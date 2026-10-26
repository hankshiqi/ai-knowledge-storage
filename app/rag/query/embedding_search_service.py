from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.infra.llm.providers import llm_provider
from app.infra.persistentce.history_repository import history_repository
def confirm_item_name_and_rewritten_query(state):
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")
    if item_names is None or rewritten_query is None:
        logger.error("item_name or rewritten_query is None")
        raise ValueError("item_name and rewritten_query must be provided")
    return item_names,rewritten_query

def search_by_milvus(item_name,rewritten_query):
    #embedding
    rewritten_query_vector=llm_provider.generate_embeddings([rewritten_query])
    #构造request
    reqs = milvus_gateway.create_hybrid_search_requests(
        dense_vector=rewritten_query_vector["dense"][0],
        sparse_vector=rewritten_query_vector["sparse"][0],
        expr = f"item_name in {item_name}",
# 如果 item_name 是 ['HAK 180 烫金机']，拼出来就是：item_name in ['HAK 180 烫金机'] （完美）
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
def search_by_embedding(state: QueryGraphState) -> QueryGraphState:
    """
    向量检索服务：
    1. 根据改写后的问题和限定的商品范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-K 最相关的知识切片
    4. 回写 embedding_chunks
    """
    #先去进行对item_name和rewritten_query进行检查
    item_name,rewritten_query=confirm_item_name_and_rewritten_query(state)
    milvus_result=search_by_milvus(item_name,rewritten_query)
    #修改格式
    embedding_chunks=deal_milvus_list(milvus_result)
    return embedding_chunks

if __name__ == "__main__":
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机使用说明",
        "item_name": ["Brother HAK 180烫金机"],
        "is_stream": False,
    }
    result = search_by_embedding(test_state)
    print(result)