import json
import math
from numbers import Real

from pymilvus import DataType
from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log

# BGE-M3 稠密向量的维度，必须与 Milvus 集合 schema 保持一致。
DENSE_VECTOR_DIM = 1024
# Milvus VARCHAR 最大长度按 UTF-8 字节数校验，而不是 Python 字符数。
STRING_FIELD_LIMITS = {
    "title": 512,
    "file_title": 512,
    "parent_title": 512,
    "item_name": 512,
    "content": 65535,
}
EXPECTED_FIELD_TYPES = {
    "chunk_id": DataType.INT64,
    "title": DataType.VARCHAR,
    "file_title": DataType.VARCHAR,
    "parent_title": DataType.VARCHAR,
    "item_name": DataType.VARCHAR,
    "part": DataType.INT64,
    "content": DataType.VARCHAR,
    "dense_vector": DataType.FLOAT_VECTOR,
    "sparse_vector": DataType.SPARSE_FLOAT_VECTOR,
}


def _get_client():
    """获取可用的 Milvus 客户端，避免后续调用在 None 上失败。"""
    client = milvus_gateway.milvus_client
    if client is None:
        raise ConnectionError("Milvus client is unavailable; check MILVUS_URL and server connectivity")
    return client

@step_log("_validate_existing_collection")
def _validate_existing_collection(client, collection_name: str):
    """检查已有集合的字段、主键和向量维度是否符合当前导入格式。"""
    description = client.describe_collection(collection_name)
    fields = {field["name"]: field for field in description.get("fields", [])}
    required_fields = set(EXPECTED_FIELD_TYPES)
    missing_fields = sorted(required_fields - fields.keys())
    if missing_fields:
        raise RuntimeError(
            f"Milvus collection {collection_name!r} is missing fields {missing_fields}; "
            "migrate or recreate the collection before importing"
        )
    if "file_hash" not in fields and description.get("enable_dynamic_field") is not True:
        raise RuntimeError(
            f"Milvus collection {collection_name!r} cannot store file_hash; "
            "enable dynamic fields or recreate the collection with a file_hash field"
        )
    if description.get("auto_id") is not True:
        raise RuntimeError(
            f"Milvus collection {collection_name!r} does not use auto-generated IDs, "
            "but the importer omits chunk_id"
        )
    for name, expected_type in EXPECTED_FIELD_TYPES.items():
        actual_type = fields[name].get("type")
        if actual_type is None or int(actual_type) != int(expected_type):
            raise RuntimeError(
                f"Milvus collection {collection_name!r} field {name!r} has type "
                f"{actual_type}, expected {expected_type}"
            )
    if fields["chunk_id"].get("is_primary") is not True:
        raise RuntimeError(f"Milvus collection {collection_name!r} chunk_id is not the primary key")
    for name, expected_length in STRING_FIELD_LIMITS.items():
        actual_length = fields[name].get("params", {}).get("max_length")
        if actual_length is None or int(actual_length) < expected_length:
            raise RuntimeError(
                f"Milvus collection {collection_name!r} field {name!r} max_length is "
                f"{actual_length}, expected at least {expected_length}"
            )

    dense_dim = fields["dense_vector"].get("params", {}).get("dim")
    if dense_dim is None or int(dense_dim) != DENSE_VECTOR_DIM:
        raise RuntimeError(
            f"Milvus collection {collection_name!r} dense_vector dimension is {dense_dim}, "
            f"but the importer produces {DENSE_VECTOR_DIM}"
        )
@step_log("check_embeddings_content")
def check_embeddings_content(state: ImportGraphState):
    """在访问集合前确认待导入切片存在且非空。"""
    embedding_content = state.get('embedding_content')
    if embedding_content is None:
        logger.error('embedding_content is None')
        raise ValueError('embedding_content is None')
    if len(embedding_content) == 0:
        logger.error('embedding_content is empty')
        raise ValueError('embedding_content is empty')
    return embedding_content
@step_log("validate_embeddings_content")
def validate_embeddings_content(state: ImportGraphState, embedding_content: list[dict]):
    """补齐可从流程状态推导的字段，并在删除旧数据前校验全部切片。"""
    file_title = state.get("file_title")
    item_name = state.get("item_name")
    file_hash = state.get("file_hash")
    if not isinstance(file_title, str) or not file_title.strip():
        raise ValueError("file_title is required before importing chunks")
    if not isinstance(item_name, str) or not item_name.strip():
        raise ValueError("item_name is required before importing chunks")
    if not isinstance(file_hash, str) or len(file_hash) != 64:
        raise ValueError("file_hash must be a SHA-256 hex digest before importing chunks")

    for index, chunk in enumerate(embedding_content):
        if not isinstance(chunk, dict):
            raise TypeError(f"embedding_content[{index}] must be a dict")

        # 文件名和商品名来自流程状态，确保每条切片都能按文件归属检索。
        if not chunk.get("file_title"):
            chunk["file_title"] = file_title
        if chunk["file_title"] != file_title:
            raise ValueError(f"embedding_content[{index}] has a different file_title")
        chunk["item_name"] = item_name
        # file_hash is a dynamic field in the chunks collection. It is the
        # document identity used for replacement; file_title is only display data.
        chunk["file_hash"] = file_hash
        if not chunk.get("parent_title"):
            chunk["parent_title"] = chunk.get("title") or ""
        if chunk.get("part") is None:
            chunk["part"] = 0

        required_fields = (*STRING_FIELD_LIMITS, "part", "dense_vector", "sparse_vector")
        missing_fields = [
            field for field in required_fields
            if field not in chunk or chunk[field] is None
        ]
        if missing_fields:
            raise ValueError(f"embedding_content[{index}] is missing fields: {missing_fields}")

        # 提前检查 Milvus 字段长度，避免部分数据写入后才因超长失败。
        for field, max_bytes in STRING_FIELD_LIMITS.items():
            value = chunk[field]
            if not isinstance(value, str):
                raise TypeError(f"embedding_content[{index}].{field} must be a string")
            if len(value.encode("utf-8")) > max_bytes:
                raise ValueError(
                    f"embedding_content[{index}].{field} exceeds Milvus max_length={max_bytes} bytes"
                )

        part = chunk["part"]
        if isinstance(part, bool) or not isinstance(part, int):
            raise TypeError(f"embedding_content[{index}].part must be an integer")
        # 拒绝维度不匹配、NaN 或无穷值，防止无效向量进入索引。
        dense_vector = chunk["dense_vector"]
        if not isinstance(dense_vector, list) or len(dense_vector) != DENSE_VECTOR_DIM:
            raise ValueError(
                f"embedding_content[{index}].dense_vector must contain {DENSE_VECTOR_DIM} values"
            )
        if not all(isinstance(value, Real) and math.isfinite(value) for value in dense_vector):
            raise ValueError(f"embedding_content[{index}].dense_vector contains an invalid value")
        if not isinstance(chunk["sparse_vector"], dict):
            raise TypeError(f"embedding_content[{index}].sparse_vector must be a dict")
        # 本项目的稀疏向量格式为“特征索引: 权重”。
        for sparse_index, sparse_value in chunk["sparse_vector"].items():
            if isinstance(sparse_index, bool) or not isinstance(sparse_index, int) or sparse_index < 0:
                raise ValueError(
                    f"embedding_content[{index}].sparse_vector has an invalid index: {sparse_index!r}"
                )
            if not isinstance(sparse_value, Real) or not math.isfinite(sparse_value):
                raise ValueError(
                    f"embedding_content[{index}].sparse_vector[{sparse_index}] must be finite numeric data"
                )

    return embedding_content

@step_log("prepare_chunks_collection")
def prepare_chunks_collection():
    """不存在时创建切片集合，已存在时先确认 schema 兼容。"""
    client = _get_client()
    collection_name = milvus_gateway.get_chunks_collection
    if not collection_name:
        raise ValueError("CHUNKS_COLLECTION must be configured")
    if collection_name in client.list_collections():
        _validate_existing_collection(client, collection_name)
        logger.info(f'集合{collection_name}已经存在,无需再次创建')
        return

    logger.info(f'集合{collection_name}不存在，创建集合')
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name='chunk_id', datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name='file_title', datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name='parent_title', datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name='item_name', datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name='part', datatype=DataType.INT64)
    schema.add_field(field_name='content', datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=DENSE_VECTOR_DIM)
    schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        metric_type="COSINE",
        params={"M": 64, "efConstruction": 100}
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_bm25_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f'集合{collection_name}创建成功')
@step_log("delete_chunks_by_file_hash")
def delete_chunks_by_file_hash(state: ImportGraphState):
    """按内容哈希删除同一版本的旧切片，避免同名文件互相覆盖。"""
    client = _get_client()
    collection_name = milvus_gateway.get_chunks_collection
    file_hash = state.get("file_hash")
    if not isinstance(file_hash, str) or len(file_hash) != 64:
        raise ValueError("file_hash must be a SHA-256 hex digest before deleting existing chunks")

    # 使用 JSON 字符串字面量构造过滤表达式，正确转义哈希值。
    hash_literal = json.dumps(file_hash, ensure_ascii=False)
    delete_res = client.delete(
        collection_name=collection_name,
        filter=f"file_hash == {hash_literal}"
    )
    logger.info(f"已删除 file_hash 为 '{file_hash}' 的数据，数量: {delete_res.get('delete_count', 0)}")
    return state # 修复：必须返回 state
@step_log("insert_chunks_into_db")
def insert_chunks_into_db(state: ImportGraphState, embedding_content: list[dict]):
    """写入切片并将 Milvus 自动生成的主键回填到流程状态。"""
    client = _get_client()
    collection_name = milvus_gateway.get_chunks_collection

    # 重跑时移除上一次回填的自动主键，主键应由 Milvus 重新生成。
    for chunk in embedding_content:
        chunk.pop("chunk_id", None)

    insert_res = client.insert(
        collection_name=collection_name,
        data=embedding_content
    )

    insert_count = insert_res.get('insert_count', 0)
    generated_ids = insert_res.get('ids', [])
    if insert_count != len(embedding_content) or len(generated_ids) != len(embedding_content):
        raise RuntimeError(
            f"Milvus reported insert_count={insert_count} and {len(generated_ids)} IDs "
            f"for {len(embedding_content)} chunks"
        )

    for i, chunk in enumerate(embedding_content):
        if i < len(generated_ids):
            chunk['chunk_id'] = generated_ids[i]

    state['embedding_content'] = embedding_content

    # 同步更新 chunks，保证后续节点读取到带有 Milvus 主键的数据。
    if 'chunks' in state:
        state['chunks'] = embedding_content

    logger.info(f"已成功插入 {insert_count} 条切片数据到 {collection_name}")
    return state
@step_log("index_chunks")
def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """执行切片校验、集合准备、旧数据清理和新数据写入。"""
    embedding_content = validate_embeddings_content(
        state,
        check_embeddings_content(state),
    )
    prepare_chunks_collection()

    state = delete_chunks_by_file_hash(state)
    state = insert_chunks_into_db(state, embedding_content)

    logger.info(f"{state.get('file_title')} 已经入库完成!")
    return state
