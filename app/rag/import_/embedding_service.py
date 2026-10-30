import json
from pathlib import Path
from typing import Any

from agents import result


from app.process.import_.agent.state import ImportGraphState
from app.rag.import_ import item_name_service
from app.rag.import_.config import CHUNKS_BATCH_SIZE
from app.infra.llm.providers import llm_provider
from app.shared.runtime.logger import logger, step_log
#检查是否完整
@step_log("get_data_and_validate")
def get_data_and_validate(state: ImportGraphState):
    item_name = state.get("item_name")
    chunks = state.get("chunks")
    md_path = state.get("md_path")

    if not item_name:
        logger.error("item_name 不能为空，请先执行商品名称识别节点")
        raise ValueError("item_name 不能为空，请先执行商品名称识别节点")

    if not chunks:
        if not md_path:
            logger.error("chunks 为空时必须提供 md_path")
            raise ValueError("chunks 为空时必须提供 md_path")
        chunks_path = Path(md_path).with_name(
            f"{Path(md_path).stem}_chunks.json"
        )
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    return chunks, item_name
#按照一定的batchsize去批量进行向量化，生成对应的稀疏和稠密向量
@step_log("batch_generate_embeddings")
def batch_generate_embeddings(chunks: list[dict[str, Any]],item_name: str,batch_size: int = CHUNKS_BATCH_SIZE) -> list[dict[str, Any]]:
    for index in range(0,len(chunks),batch_size):
        batch_chunks = chunks[index:index+batch_size]
        batch_content_list=[]
        for i in range(len(batch_chunks)):
            chunk = batch_chunks[i]
            chunk['item_name'] = item_name
            chunk['content'] = f"主体:{item_name}, 内容:{chunk['content']}"
            batch_content_list.append(chunk['content'])
        #注意这里要先去获取这一批数据的content 之后再进行批量处理
        # Chunk vectors are persisted in Milvus immediately; caching every
        # unique chunk in Mongo would add write amplification to imports.
        result = llm_provider.generate_embeddings(batch_content_list, use_cache=False)
        for i,chunk in enumerate(batch_chunks):
            chunk['dense_vector'] = result['dense'][i]
            chunk['sparse_vector'] = result['sparse'][i]
    logger.info(f'向量化完成，共处理 {len(chunks)} 个 chunk')
    return chunks
@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """

    chunks,item_name =get_data_and_validate(state)
    embedded_chunks=batch_generate_embeddings(chunks,item_name,CHUNKS_BATCH_SIZE)
    state['embedding_content']=embedded_chunks
    return state    
