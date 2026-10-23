from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.rag.query.config import NONE_RRF_K,NODE_RRF_LIMIT_TOP
def get_data_and_validates(state):
    embedding_chunks = state.get("embedding_chunks",[])
    hyde_embedding_chunks = state.get("hyde_embedding_chunks",[])
    if not len(embedding_chunks)>0 or not len(hyde_embedding_chunks)>0:
        logger.error(f"嵌入后的chunks没有导入到state！异常终止！")
        raise ValueError("嵌入后的chunks没有导入到state！异常终止！")
    return embedding_chunks,hyde_embedding_chunks


def rrf_process(rrf_list:list,topk:int=NODE_RRF_LIMIT_TOP,k:int=NONE_RRF_K):
    """
    rrf_list: [(1.0,embedding_chunks),(1.0,hyde_embedding_chunks)]
    """
    score_dict:dict[str,float]={}
    chunk_dict:dict[str,list]={}
    for weight,chunk_list in rrf_list:
        for rank,chunk in enumerate(chunk_list,start=1):
            chunk_id = chunk.get("chunk_id")
            score_dict[chunk_id] = score_dict.get(chunk_id,0.0) + weight * (1/(rank+k))
            chunk_dict.setdefault(chunk.get("chunk_id"),chunk)
    #更新分数
    new_chunk_list = []
    for chunk_id, chunk in chunk_dict.items():
        new_chunk_list.append({**chunk, "score": score_dict[chunk_id]})
    #排序
    new_chunk_list.sort(key=lambda x:x["score"],reverse=True)
    return new_chunk_list[:topk]

 
def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF 融合服务：
    1. 合并来自不同检索源的文档列表
    2. 应用 RRF 算法消除分数差异
    3. 给出综合排名最高的文档列表（Top 10）
    4. 回写 rrf_chunks
    """
    #get_data_and_validates()
    embedding_chunks,hyde_embedding_chunks=get_data_and_validates(state)
    #构造rrf_list
    rrf_list = [(1.0,embedding_chunks),(1.0,hyde_embedding_chunks)]
    #进行rrf处理
    rrf_chunks = rrf_process(rrf_list)
    #更新rrf_chunks
    state["rrf_chunks"]=rrf_chunks
    return state

if __name__ == "__main__":
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "Brother HAK 180烫金机怎么操作？",
        "rewritten_query": "Brother HAK 180烫金机的具体操作步骤是什么？",
        "item_name": ["Brother HAK 180烫金机"],
    }

    from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
    from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde

    emb_res = node_search_embedding(mock_state)
    hyde_res = node_search_embedding_hyde(mock_state)
    mock_state["embedding_chunks"] = emb_res.get("embedding_chunks") or []
    mock_state["hyde_embedding_chunks"] = hyde_res.get("hyde_embedding_chunks") or []

    result = fuse_by_rrf(mock_state)
    print(result)