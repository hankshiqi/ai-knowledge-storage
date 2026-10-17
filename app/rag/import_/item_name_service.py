import json
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
#1.获取并且校验参数
#    get file_title chunks md_path 
#   file_title没有去md_path当中去取 chunks没有去md_path_json当中去取
#
#
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
    item_name = chain.invoke(messages)

    if not item_name:
        item_name = file_title
        logger.warning(f'未识别出item_name，使用文件名作为item_name：{item_name}')
    else:
        logger.info(f'识别到的item_name是{item_name}')
    return item_name
@step_log("prepare_item_name_collection")
def prepare_item_name_collection():
    
    # Define server address
    SERVER_ADDR = infra_config.milvus.milvus_url

    # Create a MilvusClient instance
    client = milvus_gateway.milvus_client
    #上来先判断集合是否存在，先去获取item集合的名称
    collection_name = milvus_gateway.get_item_name_collection
    if collection_name in client.list_collections():
        logger.info(f'集合{collection_name}已经存在,无需再次创建')
        return 
    logger.info(f'集合{collection_name}不存在，创建集合')
    # Create a collection
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
    delete_res = client.delete(
        collection_name=collection_name,
        filter=f'file_title == "{file_title}"' 
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