from langchain.messages import HumanMessage

from app.infra.persistentce.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.shared.model.lm_utils import get_llm_client
from app.shared.runtime.logger import logger, step_log
from app.shared.runtime.load_prompt import load_prompt
from langchain_core.output_parsers import JsonOutputParser
from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.infra.llm.providers import llm_provider
from app.infra.persistentce.history_repository import history_repository
@step_log("get_data_and_validate")
def get_data_and_validate(state: QueryGraphState) -> QueryGraphState:
    """
    获取数据并检验
    """
    session_id = state.get("session_id")
    original_query = state.get("original_query")
    #对session_id和original_query进行有效性检验
    if not session_id or not original_query:
        logger.error("session_id or original_query is invalid")
        raise ValueError("session_id or original_query is invalid")
    return session_id, original_query

@step_log("get_recent_records_and_context")
def get_recent_records_and_context(session_id):
    #按照session_id获取最近的history_list
    history_list = history_repository.list_recent(session_id)
    #如果为空的话，则直接返回空的context
    if not history_list:
        logger.warning(f"当前会话{session_id}没有对话记录,提前跳出")
        return "无对话记录"
    #判断当前的这些列表当中是否有明确的item_name
    final_history_list = [
        item for item in history_list if len(item.get("item_names",[])) > 0
    ]
    if not final_history_list or len(final_history_list) ==0 :
        logger.warning(f"当前会话{session_id}没有有效的对话记录,提前跳出")
        return "无有效对话记录"
    #拼接context
    history_context = ""
    for index,item in enumerate(history_list,start=1):
        history_context += (f"序号{index}:{'提问:' if item.get('role') == 'user' else '回答:'}"
                            f"{item.get('rewritten_query') if item.get('role')=='user' else item.get('text')}"
                            f"关联主体:{','.join(item.get('item_names'))}\n"
                            )
    return history_context
@step_log("confirm_item_name_and_rewritten_query")
def confirm_item_name_and_rewritten_query(history_context, original_query):
    #获取模型对象
    model=get_llm_client(json_mode=True)
    #封装提示词对象
    prompt=load_prompt("rewritten_query_and_itemnames",query=original_query,history_text=history_context)
    message = [
        HumanMessage(
            content=prompt
        )
    ]
    #调用模型  注意使用JSON解析器
    json_parser = JsonOutputParser()
    llm = model | json_parser
    ans = llm.invoke(message)
    #进行后处理 从返回的字典当中取对应的rewritten_query以及item_names 如果没有
    rewritten_query = ans.get("rewritten_query")
    item_names = ans.get("item_names")
    if not rewritten_query:
        ans['rewritten_query']=original_query
    if not item_names or not isinstance(item_names, list):
        ans['item_names'] = []
    else:
        # 如果确实是 list，但里面夹杂了 None 或空字符串，顺手清理一下
        ans['item_names'] = [str(name).strip() for name in item_names if name]
    return ans
@step_log("get_recent_records_and_context")
def get_recent_records_and_context(session_id):
    # 按照 session_id 获取最近的 history_list
    history_list = history_repository.list_recent(session_id)

    if not history_list:
        logger.warning(f"当前会话{session_id}没有对话记录,提前跳出")
        return "无对话记录"

    # 是否存在至少一条带 item_names 的有效记录
    final_history_list = [
        it for it in history_list if it.get("item_names")
    ]
    if not final_history_list:
        logger.warning(f"当前会话{session_id}没有有效的对话记录,提前跳出")
        return "无有效对话记录"

    # 拼接到 history_list 全量(保持原逻辑)
    # 若只想拼有效记录,把 history_list 改为 final_history_list
    parts = []
    for index, it in enumerate(final_history_list, start=1):
        if it.get("role") == "user":
            prefix = "提问:"
            content = it.get("rewritten_query") or it.get("text") or ""
        else:
            prefix = "回答:"
            content = it.get("text") or ""
        names = ",".join(it.get("item_names") or [])
        parts.append(f"序号{index}:{prefix}{content}关联主体:{names}\n")
    return "".join(parts)


import logging

logger = logging.getLogger(__name__)
@step_log("select_item_names")
# def select_item_names(milvus_result:dict[str,list[dict]])->dict[str,list]:
#     confirm_list=[]
#     optional_list=[]
#     for item_name, vector_point_list in milvus_result.items():
#         high_score_list = [item for item in vector_point_list if item.get('score') >=0.73]
#         md_score_list  =  [item for item in vector_point_list if item.get('score') >=0.6 and item.get('score') <0.73]
#         if len(high_score_list)>0:
#             confirm_list.append(high_score_list[0])
#             continue
#         if len(md_score_list)>0:
#             optional_list.extend(md_score_list[:2])
#             continue


#     return {"confirm_list":confirm_list,"optional_list":optional_list}
def search_by_item_names(item_names):
        """
    基于候选主体名称到 Milvus 中检索相近主体。

    Args:
        item_names: 模型抽取出来的主体名称列表。

    Returns:
        dict[str, list[dict]]: 以原始主体名为键、相似主体候选列表为值的映射结果。
    """
        ans_dict = {}
        vector_dict = llm_provider.generate_embeddings(item_names)
        for index, item_name in enumerate(item_names):
            dense_vector = vector_dict["dense"][index]
            sparse_vector = vector_dict["sparse"][index]
            req = milvus_gateway.create_hybrid_search_requests(dense_vector,sparse_vector)
            milvus_result = milvus_gateway.hybrid_search(
                client = milvus_gateway.milvus_client,
                collection_name = milvus_gateway.get_item_name_collection,
                reqs = req,
                ranker_weights=(0.5,0.5),
                norm_score = True,
                output_fields=["item_name"]
            )
            batch = milvus_result[0] if milvus_result else []
            ans_dict[item_name] = [
                {
                    "item_name": (item.get("entity") or {}).get("item_name", ""),
                    "score": item.get("distance") or 0,
                }
                for item in batch
            ]
        return ans_dict

def select_item_names(milvus_result: dict[str, list[dict]]) -> dict[str, list]:
    confirm_list = []
    optional_list = []

    for item_name, vector_point_list in milvus_result.items():
        # 先按 score 降序，保证后面 [0] / [:2] 取到的确实是最高分
        vector_point_list = sorted(
            vector_point_list,
            key=lambda x: x.get("score", 0.0) or 0.0,
            reverse=True,
        )

        scores = [item.get("score") for item in vector_point_list]
        print(f"[{item_name}] 共 {len(scores)} 条候选, scores={scores}")

        high_score_list = [i for i in vector_point_list if (i.get("score") or 0) >= 0.73]
        md_score_list = [
            i for i in vector_point_list
            if 0.6 <= (i.get("score") or 0) < 0.73
        ]

        if high_score_list:
            best = high_score_list[0]
            print(f"[{item_name}] -> confirm, 取最高分 {best.get('score')}")
            confirm_list.append(best)
            continue

        if md_score_list:
            picked = md_score_list[:2]
            print(f"[{item_name}] -> optional, 取 {[p.get('score') for p in picked]}")
            optional_list.extend(picked)
            continue

        max_s = scores[0] if scores else None
        print(f"[{item_name}] -> 丢弃, 最高分 {max_s} 未达 0.6")

    print(f"汇总: confirm={len(confirm_list)}, optional={len(optional_list)}")
    return {"confirm_list": confirm_list, "optional_list": optional_list}


@step_log("apply_item_name_change")
def apply_item_name_change(state,list_dict,rewritten_query):
    #判断list——dict是否有确定的list
    #先去拿到两个list
    confirm_list = list_dict.get("confirm_list", [])
    optional_list = list_dict.get("optional_list", [])
    #如果confirm_list不为空，则直接更新state
    if len(confirm_list)>0:
        state["item_names"] = [item.get("item_name") for item in confirm_list]
        state["rewritten_query"] = rewritten_query
        if "answer" in state:
            state['answer'] = None
        return 
    if len(optional_list)>0:
        state["answer"] = f"本轮提问没有主体，但是有相似可选的：{','.join([item.get('item_name') for item in optional_list])}，请确认"
        return 
    state["answer"] = f"本轮提问没有主体，也没有相似的可选，请确认"

def history_save(state):
    history_repository.save_message(
        session_id=state.get("session_id"),
        role="user",
        text=state.get("original_query"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names",[]),
        image_urls=[],
    )
@step_log("confirm_item_name")
def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    意图确认服务：
    1. 结合历史对话提取商品名
    2. 将模糊问题改写为完整独立的精准问题
    3. 在 Milvus 向量库中进行混合搜索
    4. 根据评分高低自动对齐标准型号，或生成反问让用户手动确认
    5. 同步历史记录到 MongoDB
    """
    #对session_id和original_query进行有效性检验
    session_id, original_query=get_data_and_validate(state)
    #获取最近的有效记录并且拼接（默认十条），并且检验是否有效（含有item_names）
    history_context=get_recent_records_and_context(session_id)
    #调用模型识别item_name和重写的问题,模型按照json格式进行返回
    llm_dict= confirm_item_name_and_rewritten_query(history_context, original_query)
    item_names = llm_dict.get("item_names",[])
    list_dict={}
    if len(item_names)>0:
        #根据item_names进行搜索
        milvus_result:dict[str,list[dict]]=search_by_item_names(item_names)
        #根据打分确定两个列表，确认列表和可选列表
        list_dict = select_item_names(milvus_result)
    #更新state
    apply_item_name_change(state,list_dict,llm_dict.get("rewritten_query"))
    history_save(state)
    return state


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "HAK180 怎么用？",
        "is_stream": False,
    }
    result_state = confirm_item_name(mock_state)
    print(result_state)