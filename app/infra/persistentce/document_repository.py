"""MongoDB registry for imported knowledge-base documents.

本模块为知识库文档导入提供一份基于 MongoDB 的注册表，核心目标：

1. 去重：以文件内容的 SHA-256 作为唯一键，保证同一文件只被导入一次；
2. 并发协调：通过 `reserve` 实现"先占位、后处理"的预约语义，
   避免多个上传请求/多个 worker 同时处理同一文件；
3. 崩溃恢复：通过"租约（lease）"识别超时未更新的 pending/processing 记录，
   允许其被后续请求重新预约，避免僵尸任务永久占用；
4. 生命周期维护：通过 `update_status` 推进状态并刷新 `updated_at`（兼作心跳续约）。

并发正确性依赖以下几点，修改时务必保持：
- `file_hash` 上的唯一索引（防止并发插入产生重复记录）；
- 以 `status`（以及必要时 `updated_at`）作为更新条件实现 CAS，
  防止两个请求同时重置同一条记录；
- `update_status` 在传入 `task_id` 时按持有者过滤，防止旧任务覆盖新任务。

注意：本模块使用同步的 pymongo 客户端。若在异步框架（如 FastAPI）的事件循环
中直接调用，会阻塞事件循环，应改用 motor 或放入线程池执行。
"""
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo.errors import DuplicateKeyError

from app.shared.clients.mongo_history_utils import get_history_mongo_tool
from app.shared.runtime.logger import logger


# 存放文档导入记录的集合名
DOCUMENT_COLLECTION = "knowledge_documents"
# 预约租约默认时长（秒）：默认 24 小时。
# 超过该时长仍未更新 `updated_at` 的 pending/processing 记录被视为"僵尸"，
# 可被后续请求重新预约。
DEFAULT_RESERVATION_LEASE_SECONDS = 24 * 60 * 60
# 合法的文件哈希格式：小写十六进制 SHA-256（64 位）
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> datetime:
    """返回带 UTC 时区的当前时间。

    统一使用 aware datetime，避免与 pymongo 返回的 naive datetime
    做减法时出现 `can't subtract offset-naive and offset-aware datetimes`。
    """
    return datetime.now(timezone.utc)


class DocumentRepository:
    """Persist one import record per file hash and coordinate duplicate uploads."""

    def __init__(self) -> None:
        # 集合句柄的进程内缓存，首次访问时初始化（建索引、取 collection）
        self._collection_cache = None
        # 预约租约时长（秒），最小 60 秒，避免配置成 0 或负数导致记录瞬间过期
        self.reservation_lease_seconds = max(
            60,
            int(os.getenv("DOCUMENT_RESERVATION_LEASE_SECONDS", str(DEFAULT_RESERVATION_LEASE_SECONDS))),
        )

    @staticmethod
    def _validate_hash(file_hash: str) -> None:
        """校验 file_hash 是合法的小写 SHA-256 十六进制串，否则抛 ValueError。

        这是整个去重逻辑的根基：哈希格式不正确会导致唯一索引失去意义。
        """
        if not isinstance(file_hash, str) or not _SHA256_RE.fullmatch(file_hash):
            raise ValueError("file_hash must be a lowercase SHA-256 hex digest")

    def _is_stale(self, document: dict[str, Any], now: datetime) -> bool:
        """判断一条 pending/processing 记录是否已超过租约、可被重新预约。
    
        - 优先使用 `updated_at`，缺失时回退到 `created_at`；
        - 若时间字段不是 datetime（例如被外部写入破坏），返回 False，
          即"不视为过期"，宁可保留现状也不误判（代价是此类记录可能永久卡住，
          需要运维介入清理）；
        - naive datetime 按 UTC 处理，与 pymongo 默认行为保持一致。
        """
        updated_at = document.get("updated_at") or document.get("created_at")
        if not isinstance(updated_at, datetime):
            return False
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return now - updated_at > timedelta(seconds=self.reservation_lease_seconds)

    def _collection(self):
        """懒加载并缓存集合句柄，同时确保索引存在。

        索引说明：
        - `file_hash` 唯一索引：并发插入的最终防线，配合 reserve 中的
          DuplicateKeyError 处理，保证同一哈希只有一条记录；
        - `task_id` 普通索引：便于按任务反查文档。

        注意：此处的"检查-赋值"不是原子操作，多线程首次并发访问时可能重复
        执行 create_index（幂等，仅多一次网络往返）。如需严格保证，
        可在应用启动阶段显式初始化，或对首次初始化加锁。
        """
        if self._collection_cache is None:
            collection = get_history_mongo_tool().db[DOCUMENT_COLLECTION]
            collection.create_index("file_hash", unique=True)
            collection.create_index("task_id")
            self._collection_cache = collection
        return self._collection_cache

    def reserve(
        self,
        *,
        file_hash: str,
        file_title: str,
        task_id: str,
        original_filename: str,
        local_file_path: str,
    ) -> dict[str, Any]:
        """尝试为某个文件哈希预约一次导入，或返回已存在的预约信息。

        返回值:
            {"accepted": bool, "task_id": str, "status": str}
            - accepted=True：本次调用成功获得处理权（新建或重置旧记录），
              调用方应推进导入流程；
            - accepted=False：已有其他任务持有该记录，调用方应放弃或改为查询。

        流程：
        1. 记录不存在 —— 直接插入（并发插入由唯一索引兜底）；
        2. 记录存在且为 failed —— 无条件重置为 pending 并接管；
        3. 记录存在且为 pending/processing 且已过期 —— 以 `updated_at`
           做乐观锁将其重置为 pending 并接管；
        4. 其余情况 —— 返回当前持有者信息，拒绝重复预约。
        """
        self._validate_hash(file_hash)
        collection = self._collection()
        now = _utc_now()
        existing = collection.find_one({"file_hash": file_hash})

        if existing:
            status = existing.get("status", "pending")

            # 分支一：上一次导入失败，允许重新预约。
            # 以 status="failed" 作为 CAS 条件：若并发请求中已有人把状态改走，
            # matched_count 为 0，本请求即视为失败者，回查当前状态后返回。
            if status == "failed":
                result = collection.update_one(
                    {"_id": existing["_id"], "status": "failed"},
                    {
                        "$set": {
                            "file_title": file_title,
                            "original_filename": original_filename,
                            "local_file_path": local_file_path,
                            "task_id": task_id,
                            "status": "pending",
                            "error": "",
                            "updated_at": now,
                        }
                    },
                )
                if result.matched_count:
                    return {"accepted": True, "task_id": task_id, "status": "pending"}
                # CAS 失败：说明状态已被其他请求改变，回查后如实返回对方信息
                existing = collection.find_one({"file_hash": file_hash}) or {}
                return {
                    "accepted": False,
                    "task_id": existing.get("task_id", ""),
                    "status": existing.get("status", "pending"),
                }

            # 分支二：任务仍在进行中，但已超过租约，视为僵尸任务，允许抢占。
            # 由于状态值本身没变（仍是 pending/processing），单靠 status 无法
            # 防止 ABA，因此额外把 `updated_at` 的原值放进查询条件作为乐观锁：
            # 只有在这段时间内没人刷新过该记录（说明确实没人还活着）才能重置。
            if status in {"pending", "processing"} and self._is_stale(existing, now):
                result = collection.update_one(
                    {
                        "_id": existing["_id"],
                        "status": status,
                        "updated_at": existing.get("updated_at"),
                    },
                    {
                        "$set": {
                            "file_title": file_title,
                            "original_filename": original_filename,
                            "local_file_path": local_file_path,
                            "task_id": task_id,
                            "status": "pending",
                            "error": "",
                            "updated_at": now,
                        }
                    },
                )
                if result.matched_count:
                    return {"accepted": True, "task_id": task_id, "status": "pending"}
                # CAS 失败：可能刚被原持有者续约或他人抢占，回查最新状态
                existing = collection.find_one({"file_hash": file_hash}) or {}
                status = existing.get("status", "pending")

            # 分支三：记录仍被有效持有（未过期）或其他状态，拒绝重复预约，
            # 返回当前持有者信息，便于调用方展示"处理中/已完成"等提示。
            return {
                "accepted": False,
                "task_id": existing.get("task_id", ""),
                "status": status,
            }

        # 记录不存在：尝试新建。并发情况下可能多个请求同时走到这里，
        # 由 `file_hash` 唯一索引保证只有一个能插入成功。
        document = {
            "file_hash": file_hash,
            "file_title": file_title,
            "original_filename": original_filename,
            "local_file_path": local_file_path,
            "task_id": task_id,
            "status": "pending",
            "error": "",
            "created_at": now,
            "updated_at": now,
        }
        try:
            collection.insert_one(document)
            return {"accepted": True, "task_id": task_id, "status": "pending"}
        except DuplicateKeyError:
            # 竞争失败：其他请求已抢先插入，回查并返回赢家信息。
            # 极端情况下（插入后被立即删除）existing 可能为空，
            # 此时返回一个保守的 pending 状态，调用方应自行重试或查询。
            existing = collection.find_one({"file_hash": file_hash}) or {}
            return {
                "accepted": False,
                "task_id": existing.get("task_id", ""),
                "status": existing.get("status", "pending"),
            }

    def update_status(
        self,
        file_hash: str,
        status: str,
        *,
        task_id: str | None = None,
        item_name: str | None = None,
        error: str = "",
    ) -> None:
        """更新某条导入记录的状态及相关字段，并刷新 `updated_at`（兼作心跳）。

        参数:
            file_hash: 目标文档哈希。
            status:    新状态（如 processing / completed / failed）。
            task_id:   持有者任务 ID。传入时会同时写入并作为查询条件，
                       确保只有当前持有者能更新，防止旧任务覆盖新任务；
                       不传则仅按 file_hash 更新（会更新到当前任意持有者）。
            item_name: 非 None 时写入 `item_name`（允许写入空串）。
            error:     仅在非空时写入 `error`，避免正常更新误清空错误信息；
                       如需清空错误，应通过 reserve 的 failed 重置分支完成。

        注意：`matched_count == 0` 仅记录 warning，不抛异常。
        若调用方依赖"更新一定生效"，需要额外检查日志或改为返回布尔值。
        """
        values: dict[str, Any] = {"status": status, "updated_at": _utc_now()}
        if task_id:
            values["task_id"] = task_id
        if item_name is not None:
            values["item_name"] = item_name
        if error:
            values["error"] = error

        query = {"file_hash": file_hash}
        # 传入 task_id 时加持有者过滤，实现"只有 owner 能改自己的记录"
        if task_id:
            query["task_id"] = task_id

        result = self._collection().update_one(query, {"$set": values})
        if result.matched_count == 0:
            # 未匹配到记录：可能是哈希不存在、或 task_id 已易主（被抢占/重置）
            logger.warning(f"No document registry row found for file_hash={file_hash}")

    def get_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        """按文件哈希查询导入记录，不存在时返回 None。"""
        return self._collection().find_one({"file_hash": file_hash})


# 模块级单例，供全局复用（同时复用其内部缓存的 collection 句柄）
document_repository = DocumentRepository()