# 知识库文件去重与增量更新优化

## 1. 问题确认

原导入接口每次上传文件都会直接启动完整导入图：PDF/Markdown 解析、切片、商品名识别、Embedding 和 Milvus 入库。上传相同文件不会被识别为重复内容，因此会重复消耗解析和向量化资源。

原切片入库逻辑按 `file_title` 删除旧数据，商品名索引也按 `file_title` 删除旧记录。这会导致两个用户上传同名但内容不同的文件时，后一次导入删除前一次数据。

## 2. 优化后的规则

文件内容使用 SHA-256 计算 `file_hash`，文件名只作为展示字段：

| 场景 | 处理方式 |
| --- | --- |
| 相同文件内容、文件名相同或不同 | MongoDB 唯一哈希登记，跳过完整导入 |
| 文件名相同、内容不同 | 视为两个文档，分别解析并保留两份向量 |
| 同一哈希的失败任务再次上传 | 允许重新占用该哈希并重试 |
| 同一哈希正在处理 | 返回原任务 ID，不创建第二个导入任务 |
| 同一哈希已完成 | 返回原任务 ID，并提示已跳过 |

## 3. 代码实现

### 3.1 MongoDB 文档登记

新增 `knowledge_documents` 集合，由 `DocumentRepository` 管理。集合按 `file_hash` 创建唯一索引，保存：

- `file_hash`
- `file_title`、`original_filename`
- `task_id`
- `local_file_path`
- `status`：`pending`、`processing`、`completed`、`failed`
- `item_name`、`error`、创建和更新时间

上传完成后先写入登记表，再提交后台任务。唯一索引可以让并发上传同一文件最终只保留一个任务。
`pending`/`processing` 记录带有租约；默认 24 小时没有更新时视为异常中断并允许下一次上传接管，时间可通过
`DOCUMENT_RESERVATION_LEASE_SECONDS` 调整。超过该时长的超长导入应配合状态心跳或适当延长租约，避免被误接管。

### 3.2 切片向量按哈希替换

`file_hash` 被写入 `kb_chunks` 的动态字段。入库前按：

```text
file_hash == "<sha256>"
```

删除同一版本的旧切片，再插入新切片。`file_title` 不再承担数据唯一标识，因此同名文件不会互相覆盖。

### 3.3 商品名索引兼容处理

新建商品名集合时增加 `file_hash` 字段，并按哈希删除旧记录。已有旧集合如果没有 `file_hash` 字段，则不再按 `file_title` 删除任何记录，只追加新记录；这样可以先保证同名文件不被覆盖。

已有集合如需完整支持商品名索引的按哈希更新，应重建 `ITEM_NAME_COLLECTION` 集合，使其包含：

```text
file_hash VARCHAR(64)
```

重建后重新导入文档即可补齐哈希字段。

## 4. 相关文件

- `app/shared/utils/file_hash.py`：流式计算 SHA-256。
- `app/infra/persistentce/document_repository.py`：MongoDB 文档登记与并发去重。
- `app/api/http/import_server.py`：上传哈希计算、重复上传短路和任务状态同步。
- `app/process/import_/agent/state.py`：在导入状态中传递 `file_hash`。
- `app/rag/import_/index_service.py`：切片按哈希删除和写入动态字段。
- `app/rag/import_/item_name_service.py`：商品名索引按哈希替换及旧集合兼容。

## 5. 旧数据处理

历史 `kb_chunks` 数据没有 `file_hash`，无法可靠地反推出原始文件内容。优化上线后新导入的数据会带哈希；历史数据需要根据原始文件重新导入，或者单独执行一次离线回填。回填完成前，不应把历史数据与新数据混用来判断同一文件是否重复。

如果旧商品名集合没有 `file_hash` 字段，推荐在低峰期重建集合并重新导入；代码在迁移完成前会避免按文件名删除，优先保证同名文件不丢失。

## 6. 验证

已验证：

- SHA-256 采用分块读取，不会一次性把大文件加载进内存；
- 导入代码会在 `kb_chunks` 支持动态字段时写入并按 `file_hash` 过滤；不支持动态字段的旧集合会在导入前明确报迁移错误；
- 新增代码通过 Python 语法检查；
- 同哈希并发登记依赖 MongoDB 唯一索引收敛到单一任务。

## 7. 当前边界

文件登记和去重记录已经持久化到 MongoDB，但导入任务的 SSE/节点进度仍由 `task_utils.py` 的进程内字典维护。
因此服务重启后，重复文件仍能通过哈希被识别，但旧任务的实时进度和完成通知仍可能丢失；这属于任务状态持久化改造，当前未在本次去重改动中处理。
