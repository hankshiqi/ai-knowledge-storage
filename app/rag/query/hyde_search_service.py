from agents import Model
from langchain.messages import HumanMessage

from app.process.query.agent.state import QueryGraphState

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.infra.llm.providers import llm_provider
from app.infra.persistentce.history_repository import history_repository
from app.shared.runtime.load_prompt import load_prompt
from langchain_core.output_parsers import StrOutputParser
def confirm_item_name_and_rewritten_query(state):
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")
    if item_names is None or rewritten_query is None:
        logger.error("item_name or rewritten_query is None")
        raise ValueError("item_name and rewritten_query must be provided")
    return item_names,rewritten_query

def search_by_milvus(item_name,rewritten_query,answer):
    #embedding
    rewritten_query_vector=llm_provider.generate_embeddings([rewritten_query+":"+answer])
    #构造request
    reqs = milvus_gateway.create_hybrid_search_requests(
        dense_vector=rewritten_query_vector["dense"][0],
        sparse_vector=rewritten_query_vector["sparse"][0],
        expr = f"item_name in {item_name}",
        limit = 2*5
    )
    #search
    milvus_result=milvus_gateway.hybrid_search(
        client = milvus_gateway.milvus_client,
        collection_name = milvus_gateway.get_chunks_collection,
        reqs = reqs,
        ranker_weights=(0.6,0.4),  #密集的权重更大
        norm_score = True,
        limit = 5,
        output_fields = [
            "chunk_id",
            "file_title",
            "title",
            "parent_title",
            "part",
            "item_name",
            "content"
        ]
    )
    return milvus_result[0] if milvus_result and len(milvus_result)>0 else []
def deal_milvus_list(milvus_result):
    embedding_chunks=[]
    for item in milvus_result:
        entity = item.get("entity",{})
        embedding_chunks.append({
            "chunk_id":entity.get("chunk_id"),
            "score":item.get("distance",0.0),
            "file_title":entity.get("file_title"),
            "title":entity.get("title"),
            "parent_title":entity.get("parent_title"),
            "part":entity.get("part"),
            "item_name":entity.get("item_name"),
            "content":entity.get("content"),
            "source":"milvus",
            "url":""
        })
    return embedding_chunks
def llm_enrich(rewritten_query):
    #获取大模型
    Model = llm_provider.chat()
    #封装提示词
    prompt=load_prompt("hyde_prompt",rewritten_query=rewritten_query)
    message=[
        HumanMessage(content=prompt)
    ]
    llm = Model | StrOutputParser()
    llm_query=llm.invoke(message)
    return llm_query
def search_by_hyde(state: QueryGraphState) -> QueryGraphState:
    """
    HyDE 检索服务：
    1. 让 LLM 基于问题虚构一个"理想答案"
    2. 对这个假设性答案进行向量化
    3. 用答案向量在 Milvus 中检索真实文档
    4. 回写 hyde_embedding_chunks
    """
    #先去进行对item_name和rewritten_query进行检查
    item_name,rewritten_query=confirm_item_name_and_rewritten_query(state)
    #使用llm生成假设性回答
    llm_answer=llm_enrich(rewritten_query)
    milvus_result=search_by_milvus(item_name,rewritten_query,llm_answer)
    #修改格式
    hyde_embedding_chunks=deal_milvus_list(milvus_result)
    return hyde_embedding_chunks


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "Brother HAK 180烫金机怎么操作？",
        "rewritten_query": "Brother HAK 180烫金机的具体操作步骤是什么？",
        "item_name": ["Brother HAK 180烫金机"],
        "is_stream": False,
    }
    result = search_by_hyde(mock_state)
    print(result)