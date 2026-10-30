"""BGE-M3 稠密向量和稀疏向量的持久化缓存。"""

from __future__ import annotations

import hashlib
import os
import time
import unicodedata
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # 仅用于类型检查，运行时通过 _get_collection() 延迟导入 pymongo。
    from pymongo import MongoClient

from app.shared.config.embedding_config import embedding_config
from app.shared.runtime.logger import logger

# 未配置模型名称或设备时使用的默认值。
_DEFAULT_MODEL = "BAAI/bge-m3"
_DEFAULT_DEVICE = "cpu"


def normalize_embedding_text(text: str) -> str:
    """规范化不会改变语义的文本差异，尽量让相同问题命中同一个缓存键。"""
    # NFKC 可以统一全角/半角等兼容字符形式。
    # 同时统一不同平台的换行符，并去掉首尾空白。
    return unicodedata.normalize(
        "NFKC", text.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()


def make_embedding_cache_key(text: str) -> str:
    """根据当前 BGE-M3 配置和文本生成稳定的 SHA-256 缓存键。"""
    # 优先使用本地模型路径，其次使用模型名称，最后使用默认模型名称。
    model_name = embedding_config.bge_m3_path or embedding_config.bge_m3 or _DEFAULT_MODEL
    # 获取当前 embedding 使用的设备，例如 cpu 或 cuda。
    device = embedding_config.bge_device or _DEFAULT_DEVICE
    # 把会影响向量结果的配置拼接成模型指纹。
    fingerprint = "|".join(
        (
            # 手动版本号用于模型或向量策略升级时主动切换缓存空间。
            embedding_config.cache_model_version,
            # 模型路径/名称变化时，不能复用旧模型生成的向量。
            model_name,
            # 不同设备通常对应不同的运行配置。
            device,
            # FP16 设置变化时，也使用新的缓存键。
            str(bool(embedding_config.bge_fp16)),
            # 当前向量使用 L2 归一化，作为缓存策略的一部分记录。
            "normalized-l2",
        )
    )
    # 用空字符分隔模型指纹和文本，避免两部分拼接产生歧义。
    payload = f"{fingerprint}\0{normalize_embedding_text(text)}".encode("utf-8")
    # 返回固定长度的十六进制哈希字符串，作为 MongoDB 的 _id 和内存缓存 key。
    return hashlib.sha256(payload).hexdigest()


def _decode_sparse(value: Any) -> dict[int, float]:
    """把 MongoDB 中字符串键的稀疏向量恢复为 Milvus 所需的整数键。"""
    # MongoDB/BSON 对象的字段名必须是字符串，读取时需要转换回整数维度。
    if not isinstance(value, dict):
        # 缓存记录缺失或格式不正确时，返回空稀疏向量，避免直接抛出类型错误。
        return {}
    return {int(index): float(weight) for index, weight in value.items()}


class MongoEmbeddingCache:
    """带进程内 LRU 和 MongoDB 持久化的最佳努力缓存。"""

    def __init__(self) -> None:
        # MongoClient 延迟初始化，避免导入模块时立即连接数据库。
        self._client: "MongoClient | None" = None
        # MongoDB 集合句柄，首次真正访问缓存时才创建。
        self._collection = None
        # 同时保护 MongoDB 初始化和进程内 LRU 的读写。
        self._lock = RLock()
        # MongoDB 故障后的熔断截止时间，避免每次请求都等待连接超时。
        self._disabled_until = 0.0
        # key -> (过期时间戳, 向量值)，按最近使用顺序保存进程内缓存。
        self._memory: OrderedDict[str, tuple[float | None, dict[str, Any]]] = OrderedDict()

    def _mark_unavailable(self, error: Exception) -> None:
        # MongoDB 不可用时暂时熔断 30 秒，避免每次查询都重复等待网络超时。
        self._disabled_until = time.monotonic() + 30.0
        # 缓存是可选优化；记录告警后，调用方仍可继续执行模型编码。
        logger.warning(f"Embedding cache unavailable; continuing without cache: {error}")

    def _get_collection(self):
        # 缓存关闭或仍处于故障熔断期时，不访问 MongoDB。
        if not embedding_config.cache_enabled or time.monotonic() < self._disabled_until:
            return None
        # 已经初始化过集合句柄时直接复用，避免重复创建客户端和集合对象。
        if self._collection is not None:
            return self._collection

        # 多线程下只允许一个线程完成 MongoDB 客户端和集合初始化。
        with self._lock:
            if self._collection is not None:
                return self._collection
            try:
                # 延迟导入 pymongo，让缓存依赖不会阻碍不使用缓存的场景启动。
                from pymongo import MongoClient

                # 读取现有项目的 MongoDB 连接配置。
                mongo_url = os.getenv("MONGO_URL")
                db_name = os.getenv("MONGO_DB_NAME")
                if not mongo_url or not db_name:
                    raise RuntimeError("MONGO_URL and MONGO_DB_NAME are required for embedding cache")
                # 使用较短超时，避免缓存服务异常拖慢查询主流程。
                timeout = embedding_config.cache_server_selection_timeout_ms
                self._client = MongoClient(
                    mongo_url,
                    serverSelectionTimeoutMS=timeout,
                    connectTimeoutMS=timeout,
                    socketTimeoutMS=timeout,
                )
                # 根据配置获取 embedding 缓存集合；MongoDB 会在首次写入时创建集合。
                collection = self._client[db_name][embedding_config.cache_collection]
                if embedding_config.cache_ttl_seconds > 0:
                    # 配置 TTL 时，让 MongoDB 按 expires_at 自动清理过期缓存。
                    collection.create_index("expires_at", expireAfterSeconds=0, sparse=True)
                # 保存集合句柄，后续读写直接复用。
                self._collection = collection
                return collection
            except Exception as error:  # pragma: no cover - deployment dependent
                # 初始化失败时不阻断 embedding，只进入短暂熔断状态。
                self._mark_unavailable(error)
                return None

    def get_many(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        """按缓存键批量读取向量，先查进程内 LRU，再查 MongoDB。"""
        # 缓存关闭或没有待查询的 key 时，直接返回空结果。
        if not embedding_config.cache_enabled or not keys:
            return {}
        # 第一层：读取当前进程的内存 LRU，避免频繁访问 MongoDB。
        with self._lock:
            # 使用 Unix 时间戳判断进程内缓存是否过期。
            now = time.time()
            # 保存当前批次已命中的 key -> 向量结果。
            result: dict[str, dict[str, Any]] = {}
            # dict.fromkeys 用于去除同一批请求中的重复 key，同时保持原顺序。
            for key in dict.fromkeys(keys):
                # 从 OrderedDict 中取出 (过期时间, 向量值)。
                memory_entry = self._memory.get(key)
                if memory_entry is None:
                    # 内存未命中，稍后再去 MongoDB 查询。
                    continue
                expires_at, value = memory_entry
                if expires_at is not None and expires_at <= now:
                    # 内存记录已过期，删除后不返回这条旧向量。
                    self._memory.pop(key, None)
                    continue
                # 命中后移动到队尾，表示它是最近使用的记录。
                self._memory.move_to_end(key)
                # 把命中的向量加入本次读取结果。
                result[key] = value
        # 计算还没有在进程内存中命中的 key，准备查询 MongoDB。
        missing_keys = [key for key in dict.fromkeys(keys) if key not in result]
        if not missing_keys:
            # 全部命中内存时，不需要访问 MongoDB。
            return result
        # 第二层：获取 MongoDB 集合句柄。
        collection = self._get_collection()
        if collection is None:
            # MongoDB 不可用时，返回已经命中的内存结果，未命中项由上层重新编码。
            return result
        try:
            # 使用 $in 一次性查询本批次所有未命中的缓存键。
            query: dict[str, Any] = {"_id": {"$in": missing_keys}}
            if embedding_config.cache_ttl_seconds > 0:
                # 启用 TTL 时，只接受尚未过期的记录；没有 expires_at 的旧记录仍兼容读取。
                query["$or"] = [
                    {"expires_at": {"$gt": datetime.now(timezone.utc)}},
                    {"expires_at": {"$exists": False}},
                ]
            # 执行批量查询；返回的每个 document 对应一个缓存向量。
            documents = collection.find(query)
            for document in documents:
                # dense 必须是列表，否则认为缓存记录格式无效并跳过。
                dense = document.get("dense")
                if not isinstance(dense, list):
                    continue
                # MongoDB 的 _id 就是此前生成的缓存 key。
                key = str(document["_id"])
                # 读取稠密向量，并把 MongoDB 中的稀疏向量键恢复为整数。
                value = {
                    "dense": [float(value) for value in dense],
                    "sparse": _decode_sparse(document.get("sparse")),
                }
                # 把 MongoDB 命中的结果加入本次返回值。
                result[key] = value
                # 读取持久化的过期时间，供进程内 LRU 后续判断。
                expires_at = document.get("expires_at")
                expires_timestamp = expires_at.timestamp() if isinstance(expires_at, datetime) else None
                # 将 MongoDB 命中项回填到进程内 LRU，下一次优先从内存返回。
                with self._lock:
                    self._memory[key] = (expires_timestamp, value)
                    # 标记为最近使用。
                    self._memory.move_to_end(key)
                    # 超过内存上限时删除最久未使用的记录。
                    while len(self._memory) > embedding_config.cache_max_entries:
                        self._memory.popitem(last=False)
            return result
        except Exception as error:  # pragma: no cover - deployment dependent
            # MongoDB 查询失败时保留已有内存命中，并让上层继续走模型编码。
            self._mark_unavailable(error)
            return result

    def set_many(self, entries: dict[str, dict[str, Any]]) -> None:
        """批量写入 MongoDB，并更新当前进程内的 LRU。"""
        # 缓存关闭或没有待写入项时，不执行任何操作。
        if not embedding_config.cache_enabled or not entries:
            return
        # 尝试获取 MongoDB 集合；获取失败时仍会继续更新进程内 LRU。
        collection = self._get_collection()
        # 本批次所有记录共用同一个写入时间。
        now = datetime.now(timezone.utc)
        # TTL 大于 0 时计算统一过期时间；等于 0 时永久保存。
        expires_at = (
            now + timedelta(seconds=embedding_config.cache_ttl_seconds)
            if embedding_config.cache_ttl_seconds > 0
            else None
        )
        # 逐个处理 key -> 向量值。
        for key, value in entries.items():
            # 组装 MongoDB 文档；key 作为唯一 _id，value 中保存 dense/sparse 向量。
            document = {
                "_id": key,
                # 将向量元素统一转换为 Python float，便于 BSON 序列化。
                "dense": [float(item) for item in value["dense"]],
                # BSON 对象键必须是字符串，而 Milvus 的稀疏向量维度使用整数。
                "sparse": {
                    # MongoDB 对象键必须是字符串，读取时由 _decode_sparse 转回整数。
                    str(index): float(weight)
                    for index, weight in value["sparse"].items()
                },
                # 保存最后更新时间，便于运维查看缓存新鲜度。
                "updated_at": now,
            }
            if expires_at is not None:
                # 只有启用 TTL 时才写入 expires_at 字段。
                document["expires_at"] = expires_at
            if collection is not None:
                try:
                    # 按 _id 覆盖写入；不存在时通过 upsert 自动创建记录。
                    collection.replace_one({"_id": key}, document, upsert=True)
                except Exception as error:  # pragma: no cover - deployment dependent
                    # MongoDB 写失败后停止本批次后续 Mongo 写入，但仍保留进程内缓存能力。
                    self._mark_unavailable(error)
                    collection = None
            # 将 datetime 过期时间转换为内存缓存使用的 Unix 时间戳。
            expires_timestamp = expires_at.timestamp() if expires_at is not None else None
            # 第二层写入当前进程的 LRU；即使 MongoDB 暂时不可用，也能在本进程内命中。
            with self._lock:
                self._memory[key] = (
                    expires_timestamp,
                    # 复制列表和字典，避免调用方后续修改原对象影响缓存内容。
                    {"dense": list(document["dense"]), "sparse": dict(value["sparse"])},
                )
                # 写入后移动到队尾，标记为最近使用。
                self._memory.move_to_end(key)
                # 超过配置的内存条目上限时，删除队首最久未使用的记录。
                while len(self._memory) > embedding_config.cache_max_entries:
                    self._memory.popitem(last=False)


# 创建全局单例，整个进程复用同一个 Mongo 客户端、集合句柄和 LRU 缓存。
embedding_cache = MongoEmbeddingCache()
