# Embedding 向量缓存优化说明

## 1. 问题背景

查询流程会先得到 `rewritten_query`，再调用 BGE-M3 生成稠密向量和稀疏向量。原实现每次查询都会直接执行模型编码，因此热点问题、重复提问或同一请求中的重复文本会重复消耗模型计算资源。

本次优化在 embedding 层增加 MongoDB 持久化缓存，并配合进程内 LRU 缓存。相同文本在模型配置不变时可以直接返回已有的 `dense` 和 `sparse` 向量，查询流程不需要修改调用方式。

## 2. 方案概览

### 2.1 两级缓存

缓存查找顺序如下：

```text
generate_embeddings(texts)
        |
        v
进程内 LRU 缓存
        |
        | 未命中
        v
MongoDB embedding_cache 集合
        |
        | 未命中
        v
BGE-M3 编码
        |
        v
写入 MongoDB 和进程内 LRU 缓存
```

- **进程内 LRU**：减少同一服务进程内的 MongoDB 访问，默认最多保存 1024 条记录。
- **MongoDB**：在多个 API 进程或服务重启后复用向量，保存完整的稠密和稀疏向量。
- **故障降级**：MongoDB 不可用时，缓存层记录一次短暂熔断状态，embedding 继续直接调用 BGE-M3，不影响查询主流程。

当前实现选择 MongoDB，是因为项目已经使用 MongoDB 保存文档登记和任务数据；Redis 可以作为后续替换的存储实现，但不属于本次改动范围。

### 2.2 缓存键

缓存键不是原始文本本身，而是以下内容的 SHA-256：

```text
cache_model_version
模型路径或模型名称
运行设备
是否启用 FP16
向量归一化方式
\0
规范化后的文本
```

文本在计算键之前会执行以下规范化：

- 使用 Unicode NFKC 规范化；
- 将 `CRLF` 和 `CR` 统一为 `LF`；
- 去除首尾空白。

把模型配置纳入缓存键可以避免更换模型、设备或归一化策略后错误复用旧向量。升级模型时也可以直接修改 `EMBEDDING_CACHE_MODEL_VERSION`，让新请求自然生成新缓存键。

### 2.3 MongoDB 文档结构

集合名称由 `EMBEDDING_CACHE_COLLECTION` 配置，默认是 `embedding_cache`。每条记录的结构如下：

```json
{
  "_id": "<sha256 cache key>",
  "dense": [0.0123, -0.0456],
  "sparse": {"12": 0.31, "847": 0.08},
  "updated_at": "<UTC datetime>",
  "expires_at": "<UTC datetime, optional>"
}
```

MongoDB 的 BSON 对象键必须是字符串，因此稀疏向量写入时会把维度转换为字符串，读取时再转换回 Milvus 使用的整数维度。

## 3. 代码实现

- `app/shared/model/embedding_cache.py`
  - 负责文本规范化、缓存键生成、MongoDB 集合初始化、批量读取和写入；
  - 维护进程内 LRU；
  - 创建 `expires_at` 的 TTL 索引；
  - MongoDB 连接、查询或写入异常时降级为无缓存执行。
- `app/shared/model/embedding_utils.py`
  - `generate_embeddings()` 默认启用缓存；
  - 先批量查询缓存，只对未命中的文本调用 BGE-M3；
  - 同一批次中的重复文本只编码一次，再按原输入顺序返回结果；
  - 通过进程内锁二次检查，减少并发请求同时 miss 时的重复编码。
- `app/infra/llm/providers.py`
  - `LLMProvider.generate_embeddings()` 透传 `use_cache` 参数。
- `app/rag/import_/embedding_service.py`
  - 文档切片导入使用 `use_cache=False`；
  - 导入切片通常是大量一次性文本，把每个唯一切片写入缓存会产生大量 MongoDB 写放大，因此导入仍直接批量编码并将结果写入 Milvus。

调用方式：

```python
# 查询场景，默认使用缓存
result = llm_provider.generate_embeddings([rewritten_query])

# 明确绕过缓存，例如文档导入
result = llm_provider.generate_embeddings(texts, use_cache=False)
```

返回结构保持不变：

```python
{
    "dense": [[...], [...]],
    "sparse": [{123: 0.4}, {456: 0.2}],
}
```

## 4. 配置项

配置示例位于 `.env.example`：

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `EMBEDDING_CACHE_ENABLED` | `True` | 是否启用缓存；关闭后所有请求直接执行模型编码 |
| `EMBEDDING_CACHE_COLLECTION` | `embedding_cache` | MongoDB 缓存集合名称 |
| `EMBEDDING_CACHE_TTL_SECONDS` | `0` | 缓存有效期；`0` 表示不启用过期时间 |
| `EMBEDDING_CACHE_SERVER_SELECTION_TIMEOUT_MS` | `500` | MongoDB 服务发现、连接和 socket 超时时间 |
| `EMBEDDING_CACHE_MODEL_VERSION` | `v1` | 模型或向量策略版本标识，升级模型时应递增 |
| `EMBEDDING_CACHE_MAX_ENTRIES` | `1024` | 每个进程内存 LRU 的最大条目数 |

MongoDB 连接使用现有配置：

```dotenv
MONGO_URL=mongodb://127.0.0.1:27017
MONGO_DB_NAME=enterprise_rag
```

生产环境建议为缓存集合配置合适的 MongoDB 权限，并根据查询量和向量维度评估磁盘空间。启用 TTL 时，代码会在首次访问集合时创建 `expires_at` 的 TTL 索引。

## 5. 失效、升级与降级策略

1. **文本变化**：规范化后的文本不同，会生成不同缓存键。
2. **模型变化**：模型路径、设备、FP16 设置或 `EMBEDDING_CACHE_MODEL_VERSION` 变化，会生成不同缓存键，不会复用旧向量。
3. **定时失效**：`EMBEDDING_CACHE_TTL_SECONDS` 设置为正数时，MongoDB 根据 `expires_at` 自动清理过期记录；进程内缓存读取时也会检查过期时间。
4. **缓存关闭**：设置 `EMBEDDING_CACHE_ENABLED=False` 后跳过 MongoDB 和进程内缓存。
5. **MongoDB 故障**：缓存连接或读写失败时，当前请求继续使用 BGE-M3；缓存恢复后后续请求会重新尝试连接。

修改模型或向量后处理逻辑时，应优先递增 `EMBEDDING_CACHE_MODEL_VERSION`，而不是继续复用旧版本缓存。

## 6. 部署注意事项

- 查询服务的所有进程应使用同一个 `MONGO_URL`、`MONGO_DB_NAME` 和缓存集合名称，才能共享持久化缓存。
- 首次部署不会预热缓存；每个文本第一次查询时生成向量并写入缓存。
- 进程内 LRU 只在当前进程有效，服务重启后会从 MongoDB 读取已有记录。
- MongoDB 不可用时不会阻塞查询，但每次模型编码仍会产生正常的 BGE-M3 计算开销。
- 导入侧显式绕过缓存后，切片向量仍由 Milvus 持久化，不影响查询侧的 `rewritten_query` 缓存。

## 7. 验证结果

本次实现已完成以下检查：

- Python 模块编译检查（`compileall`）；
- 应用模块导入检查；
- 缓存命中后不重复调用模型的冒烟测试；
- MongoDB 稀疏向量字符串键与 Milvus 整数维度之间的序列化冒烟测试。

完整 pytest 未执行，当前虚拟环境未安装 pytest，临时安装受当前镜像网络限制。上线前应在具备 MongoDB 和 BGE-M3 模型的环境中补充多进程共享缓存、TTL 到期和 MongoDB 故障降级验证。

## 8. 当前边界

缓存按规范化后的**完全相同文本**命中，不做语义相似文本匹配；例如措辞不同但含义相近的查询仍会分别编码。缓存也不负责保存最终检索结果，因此知识库更新后不会错误复用旧的 Milvus 检索结果。
