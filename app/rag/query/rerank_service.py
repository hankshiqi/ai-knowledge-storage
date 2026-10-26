

from langchain.messages import HumanMessage

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_provider
from langchain_core.output_parsers import StrOutputParser
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS,RERANK_SUMMARY_CHAR_RATIO,RERANK_MIN_SUMMARY_CHARS,RERANK_MAX_TOPK,RERANK_MIN_TOPK,RERANK_GAP_RATIO,RERANK_GAP_ABS
def get_data_and_validate(state):
    #从state当中拿出
    rewritten_query=state.get("rewritten_query")
    rrf_chunks=state.get("rrf_chunks")
    web_search_docs=state.get("web_search_docs")
    if not rewritten_query or not len(rrf_chunks)>0 or not len(web_search_docs)>0:
        logger.error("rerank数据校验失败")
        raise Exception("rerank数据校验失败")
    return rewritten_query,rrf_chunks,web_search_docs
def get_reranker_list(rrf_chunks,web_search_docs):
    reranker_list = []
    #分别将rrf_chunks以及web_search放入得到reranker_list
    for chunk in rrf_chunks:
        reranker_list.append({
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("content"),
            "title": chunk.get("title"),
            "score":0,
            "type":"milvus",
            "url":None
        })
    for doc in web_search_docs:
        reranker_list.append({
            "chunk_id": None,
            "text": doc.get("snippet"),
            "title": doc.get("title"),
            "score":0,
            "type":"web",
            "url":doc.get("url")
        })
    return reranker_list
def get_query_answer_list(rewritten_query,reranker_list):
    #将rewritten_query向量化,tokenizer是reranker模型里面自带的
    query_answer_list=[]
    reranker_model=llm_provider.get_reranker_model()
    tokenizer = reranker_model.tokenizer
    rewritten_query_token_list=tokenizer.encode(rewritten_query,add_special_tokens=False)
    #方便计算后续的token长度，来决定是否要让模型来进行精简
    rewritten_query_token_list_len = len(rewritten_query_token_list)
    for chunk in reranker_list:
        #拿到答案的token_list
        answer = chunk["text"]
        answer_token_list = tokenizer.encode(answer, add_special_tokens=False)
        answer_token_list_len = len(answer_token_list)
        #如果答案的长度大于模型最大长度，则使用模型进行压缩
        if rewritten_query_token_list_len+answer_token_list_len+4>RERANK_MAX_INPUT_TOKENS:
            #使用模型进行压缩
            limit = max(RERANK_MIN_SUMMARY_CHARS,int(RERANK_MAX_INPUT_TOKENS-4-rewritten_query_token_list_len)/RERANK_SUMMARY_CHAR_RATIO)
            user_prompt=load_prompt("rerank_text_refine",question=rewritten_query,answer=answer,limit=limit)
            chat_model=llm_provider.chat()
            llm = chat_model | StrOutputParser()
            message = [
                HumanMessage(content=user_prompt)
            ]
            answer=llm.invoke(message)
        query_answer_list.append([rewritten_query,answer])
    return query_answer_list
def get_new_score(query_answer_list,reranker_list):
    reranker_model=llm_provider.get_reranker_model()    
    score_list=reranker_model.compute_score(query_answer_list,normalize=True)
    for score,chunk in zip(score_list,reranker_list):
        chunk["score"]=score
    #重新排序进行倒排
    reranker_list.sort(key=lambda x: x["score"], reverse=True)
def dyn_limit_rerank_docs(reranker_list):
    top_min:int=RERANK_MIN_TOPK #最小的结果限制
    top_max:int=RERANK_MAX_TOPK #最大的结果限制
    top_max = min(top_max, len(reranker_list)) #防止top_max大于列表长度造成越界
    gap_abs:float=RERANK_GAP_ABS #绝对差值
    gap_ratio:float=RERANK_GAP_RATIO #相对差值
    topk=0
    flag=0 #判断是否已经找到gap
    if top_max>top_min:
        for index in range(top_min-1,top_max-1):    
            current_score = reranker_list[index].get("score",0.0)
            next_score = reranker_list[index + 1].get("score",0.0)
            #先去判断绝对差值是否大于gap_abs
            if abs(current_score - next_score) > gap_abs:
                topk=index+1
                flag=1
                break
            #再判断相对差值是否大于gap_ratio
            if current_score>0 and (current_score - next_score) / current_score > gap_ratio:
                topk=index+1
                flag=1
                break
    if flag==0:
        topk=top_max
    return reranker_list[:topk]
    

def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF{id,title等} 和 Web Search {snippt , title,url}的文档
    2. 使用 BGE Reranker 模型计算相关性得分
    3. 根据得分动态截断，智能截取 TopK
    4. 回写 reranked_docs
    """
    #先去进行数据校验 校验rewirtten_query web_search以及rrf_chunks
    rewritten_query,rrf_chunks,web_search_docs=get_data_and_validate(state)
    #进行数据合并 得到reranker_list
    reranker_list=get_reranker_list(rrf_chunks,web_search_docs)
    #拿到query_answer_list
    query_answer_list=get_query_answer_list(rewritten_query,reranker_list)
    #利用beg-reranker模型进行重新打分
    logger.info(f"打分前的{reranker_list}")
    get_new_score(query_answer_list,reranker_list)
    logger.info(f"打分后的{reranker_list}")
    #动态截取数据
    reranker_list=dyn_limit_rerank_docs(reranker_list)
    state["reranked_docs"] = reranker_list
    return state


if __name__ == "__main__":
    mock_rewritten_query = "Python asyncio 中 gather 怎么用？如何捕获并发任务的异常？"

    # 1. 模拟内部 RRF 召回 (包含强相关、目录噪声、弱相关、完全无关)
    mock_rrf_chunks = [
        {
            "chunk_id": "local_1", 
            "title": "asyncio.gather 异常处理", 
            "content": "使用 asyncio.gather(*tasks, return_exceptions=True) 可以并发运行多个任务。若设置为 True，任务抛出的异常会作为结果返回而不会中断其他任务；若为 False，第一个异常会向外传播。"
        },
        {
            "chunk_id": "local_2", 
            "title": "目录索引", 
            "content": "第一章：异步基础\n第二章：并发控制 (gather, wait)\n第三章：异常处理"
        },
        {
            "chunk_id": "local_3", 
            "title": "asyncio.wait 用法", 
            "content": "asyncio.wait() 也是并发运行任务的方法。它返回 done 和 pending 两个集合，通常配合 timeout 参数使用，但它的异常处理机制与 gather 不同。"
        },
        {
            "chunk_id": "local_4", 
            "title": "多线程并发", 
            "content": "Python 多线程使用 threading 模块。处理 I/O 密集型任务时，使用 ThreadPoolExecutor 可以避免 GIL 限制，通过 try-except 捕获线程内的异常。"
        }
    ]

# 2. 模拟外部 Web 召回 (包含官方标准答案、实战教程、完全跑题的多进程)
    mock_web_docs = [
        {
            "title": "Python 官方文档 - 协程与任务", 
            "url": "http://docs.python.org/3/library/asyncio-task.html", 
            "snippet": "asyncio.gather(*aws, return_exceptions=False)。如果 return_exceptions 为 False，gather 会将第一个引发的异常立即传播给等待的任务。"
        },
        {
            "title": "Asyncio 实战：如何优雅地处理并发异常", 
            "url": "http://web.com/blog/1", 
            "snippet": "实战演示如何用 gather(..., return_exceptions=True) 并发请求 API。通过遍历结果并判断 isinstance(res, Exception) 来精准捕获异常，防止程序崩溃。"
        },
        {
            "title": "Python 多进程加速指南", 
            "url": "http://web.com/blog/2", 
            "snippet": "CPU 密集型任务应使用 multiprocessing。通过进程池 Pool 并发执行计算，并使用 apply_async 获取结果，这比 asyncio 更有效。"
        }
    ]
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": mock_rewritten_query,
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False,
    }
    result = rerank_documents(mock_state)
    print(result)