import json
import re
from pathlib import Path
from typing import Any

from agents import result
from attr import field
from langchain.messages import HumanMessage, SystemMessage
from pymilvus import DataType, MilvusClient

from app.infra.llm import providers
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNKS_SPLIT_TOP_NUMBER
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_provider
from langchain_core.output_parsers import StrOutputParser
from app.infra.config.providers import infra_config
from app.infra.vector_store.milvus_gateway import milvus_gateway


# ==================== 本次优化开始：商品名称识别辅助规则 ====================
# 模型可能在结果前加“商品名称：”等标签，先统一去除这些标签。
_OUTPUT_LABEL_RE = re.compile(
    r"^(?:商品名称|产品名称|item[_ -]?name)\s*[:：]\s*",
    re.IGNORECASE,
)
_CHUNK_SUFFIX_RE = re.compile(r"\s*-\d+$")
_MODEL_TOKEN_RE = re.compile(r"(?i)(?=.*[a-z])(?=.*\d)[a-z][a-z0-9]*(?:[ -]?\d+[a-z0-9]*)?")
_PRODUCT_INTRO_RE = re.compile(
    r"(?P<model>[A-Za-z]{1,}[ -]?\d+[A-Za-z0-9]*)\s*(?:是|为)\s*"
    r"(?:一款|一种|一个)?\s*(?P<product>[\u4e00-\u9fffA-Za-z]{2,20}?(?:机|器|设备))"
)
_GENERIC_TITLE_RE = re.compile(
    r"^(?:default|目录|简介|产品简介|概述|安全提示|注意事项|警告|设备|附录|图片说明|致谢)$",
    re.IGNORECASE,
)


def normalize_item_name(item_name: str) -> str:
    """在商品名称写入索引前，清理模型可能生成的格式性内容。"""
    value = str(item_name or "").strip()
    if not value:
        return ""

    # 本次优化：去除引号、代码标记、字段标签和多余空白，避免格式内容污染名称。
    value = value.strip("`\"'\u201c\u201d\u300c\u300d")
    value = _OUTPUT_LABEL_RE.sub("", value).strip()
    value = value.strip("`\"'\u201c\u201d\u300c\u300d")
    value = re.sub(r"\s+", " ", value)

    return value.strip(" \t\r\n,，。；;:：")


def extract_item_name_from_titles(chunks: list[dict[str, Any]]) -> str:
    """优先从解析器生成的标题或产品简介中提取高置信度商品名称。

    产品说明书通常会先给出型号和产品类别，再出现版权、售后和经销商信息。
    直接使用标题可以避免模型把正文中的偶然品牌提及误判为商品名称。
    """
    for chunk in chunks[:CHUNKS_SPLIT_TOP_NUMBER]:
        title = str(chunk.get("title") or "").strip()
        if not title:
            continue
        title = re.sub(r"^#+\s*", "", title)
        title = _CHUNK_SUFFIX_RE.sub("", title).strip()
        title = normalize_item_name(title)
        if not title or _GENERIC_TITLE_RE.fullmatch(title):
            continue
        if not _MODEL_TOKEN_RE.search(title):
            continue

        # 本次优化：保留标题原有写法，同时清除数字/字母与中文之间的装饰性空格。
        # 例如将“HAK 180 烫金机”规范为“HAK 180烫金机”。
        title = re.sub(r"(?<=[0-9A-Za-z])\s+(?=[\u4e00-\u9fff])", "", title)
        return title

    # 本次优化：部分说明书标题只有型号，产品类别在首段介绍中出现。
    # 例如从“HAK 180 是一款烫金机”组合出“HAK 180烫金机”。
    for chunk in chunks[:CHUNKS_SPLIT_TOP_NUMBER]:
        content = str(chunk.get("text") or chunk.get("content") or "")
        match = _PRODUCT_INTRO_RE.search(content)
        if not match:
            continue
        model = re.sub(r"\s+", " ", match.group("model")).strip()
        product = match.group("product").strip()
        return f"{model}{product}"
    return ""
# ==================== 本次优化结束：商品名称识别辅助规则 ====================
# 获取并校验参数：优先使用流程状态中的 file_title 和 chunks，缺失时从 md_path 补读。
@step_log("get_data_and_validate")
def get_data_and_validate(state: ImportGraphState):
    file_title = state.get("file_title")
    chunks = state.get("chunks")
    md_path = state.get("md_path")

    if not file_title:
        if not md_path:
            raise ValueError("file_title 和 md_path 不能同时为空")
        file_title = Path(md_path).stem

    if not chunks:
        if not md_path:
            raise ValueError("chunks 为空时必须提供 md_path")
        chunks_path = Path(md_path).with_name(
            f"{Path(md_path).stem}_chunks.json"
        )
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    return chunks, file_title
@step_log("get_item_name_from_chunks")
def get_item_name_from_chunks(chunks: list[dict[str, Any]], file_title: str) -> str:
    # 本次优化：标题或产品简介能够确定名称时直接使用，避免让模型从版权/售后文本猜测品牌。
    title_item_name = extract_item_name_from_titles(chunks)
    if title_item_name:
        logger.info(f'从产品标题确定item_name：{title_item_name}')
        return title_item_name

    # 获取到前面几个 chunks
    input_chunks = chunks[:CHUNKS_SPLIT_TOP_NUMBER]
    system_prompt = load_prompt('product_recognition_system')

    context = ''
    for chunk in input_chunks:
        title = chunk.get("title", "")
        content = chunk.get("text") or chunk.get("content", "")
        context += f"标题{title}，内容：{content}\n"

    user_prompt = load_prompt(
        'item_name_recognition',
        file_title=file_title,
        context=context
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    chat_model = llm_provider.chat()
    chain = chat_model | StrOutputParser()
    item_name = normalize_item_name(chain.invoke(messages))

    if not item_name:
        item_name = file_title
        logger.warning(f'未识别出item_name，使用文件名作为item_name：{item_name}')
    else:
        logger.info(f'识别到的item_name是{item_name}')
    return item_name
@step_log("prepare_item_name_collection")
def prepare_item_name_collection():
    
    # 获取 Milvus 服务地址配置（实际客户端由网关统一管理）。
    SERVER_ADDR = infra_config.milvus.milvus_url

    # 获取 Milvus 客户端实例。
    client = milvus_gateway.milvus_client
    #上来先判断集合是否存在，先去获取item集合的名称
    collection_name = milvus_gateway.get_item_name_collection
    if collection_name in client.list_collections():
        logger.info(f'集合{collection_name}已经存在,无需再次创建')
        return 
    logger.info(f'集合{collection_name}不存在，创建集合')
    # 创建集合 schema。
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_fields=True,
    )
    #添加对应的字段
    schema.add_field(field_name='pk',datatype=DataType.INT64,is_primary=True)
    schema.add_field(field_name='file_title',datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name='item_name',datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name='dense_vector',datatype=DataType.FLOAT_VECTOR,dim=1024)
    schema.add_field(field_name='sparse_vector',datatype=DataType.SPARSE_FLOAT_VECTOR)
    #为稠密向量和稀疏向量分别添加索引
    index_params =client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector", # Name of the vector field to be indexed
        index_type="HNSW", # Type of the index to create
        index_name="dense_vector_index", # Name of the index to create
        metric_type="COSINE", # Metric type used to measure similarity
        params={
            "M": 64, # Maximum number of neighbors each node can connect to in the graph
            "efConstruction": 100 # Number of candidate neighbors considered for connection during index construction
        } # Index building params
    )
    index_params.add_index(
        field_name="sparse_vector", # Name of the sparse vector field to index
        index_type="SPARSE_INVERTED_INDEX", # Type of the index to create
        index_name="sparse_bm25_index", # Name of the index to create
        metric_type="IP", # Metric type used for full text search
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    client.create_collection(
        collection_name=collection_name, # Name of the collection to create
        schema=schema, # Schema of the collection
        index_params=index_params, # Index parameters
    )
    logger.info(f'集合{collection_name}创建成功')

@step_log("delete_and_insert_item_name")
def delete_and_insert_item_name(item_name, file_title):
    # 1. 获取集合名称（注意加上括号调用方法）
    collection_name = milvus_gateway.get_item_name_collection 
    client = milvus_gateway.milvus_client
    
    # 2. 先删除（建议加上双引号防止特殊字符报错，并记录删除情况）
    title_literal = json.dumps(file_title, ensure_ascii=False)
    existing_rows = client.query(
        collection_name=collection_name,
        filter=f"file_title == {title_literal}",
        output_fields=["pk"],
    )
    existing_ids = [row["pk"] for row in existing_rows if row.get("pk") is not None]
    if existing_ids:
        delete_res = client.delete(
            collection_name=collection_name,
            ids=existing_ids,
        )
        logger.info(
            f"Deleted {len(existing_ids)} old item_name rows for file_title={file_title!r}; "
            f"delete_count={delete_res.get('delete_count', 0)}"
        )
    # 3. 生成向量
    result = llm_provider.generate_embeddings([item_name])
    
    # 4. 提取对应的向量
    # 注意：这里传入的是 [item_name]，所以取第 0 个元素
    dense_vec = result["dense"][0]
    sparse_vec = result["sparse"][0]
    
    # 5. 插入数据（data 必须是列表包裹字典）
    insert_res = client.insert(
        collection_name=collection_name,
        data=[
            {
                'file_title': file_title,     # 直接用函数参数
                'item_name': item_name,       # 直接用函数参数
                'dense_vector': dense_vec,    # 从字典结果中提取
                'sparse_vector': sparse_vec   # 从字典结果中提取
            }
        ]
    )    
    logger.info(f'插入 item_name: {item_name} 到集合成功！插入主键: {insert_res["ids"]}')

@step_log("recognize_and_index_item_name")
def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. 同步写入主体名称索引
    """
    chunks,file_title = get_data_and_validate(state)
    #调用LLM识别item_name
    item_name=get_item_name_from_chunks(chunks,file_title)
    #提前准备集合数据
    prepare_item_name_collection()
    #给chunks嵌入item_name
    delete_and_insert_item_name(item_name,file_title)
    state['chunks']=chunks
    state['item_name']=item_name
    return state
