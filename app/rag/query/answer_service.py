from pyexpat import model
import re
from turtle import title

from langchain.messages import HumanMessage

from app.infra.persistentce.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import add_done_task,add_running_task,push_to_session
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_provider
import time
import sys

def check_pre_answer(state: QueryGraphState) -> bool:
    pre_answer = state.get("answer")
    if not pre_answer:
        logger.info(f"没有前置答案，继续正常执行")
        return False
    #走到这里说明有答案 再去判断当前是否处于流模式当中如果是流就要现在把答案推送到session队列当中
    is_stream = state.get("is_stream")
    if is_stream:
        logger.info(f"当前处于流模式，将答案推送到session队列当中")
        push_to_session(state["session_id"], SSEEvent.DELTA, {"content": pre_answer})
    return True
def load_prompt_func(state: QueryGraphState) -> str:
    session_id = state.get("session_id")
    question = state.get("rewritten_query")

    # join 只接受一个参数;且 None 需兜底
    item_names = ",".join(state.get("item_names") or [])

    # ---- 拼接检索文档 context ----
    context_text = ""
    for index, item in enumerate(state.get("reranked_docs", []), start=1):
        title = item.get("title", "")
        source = "向量数据库" if item.get("type") == "milvus" else "网络资料"
        # score 可能是 None(存在 key 但值为 None),用 or 兜底
        score = item.get("score")
        score = 0.0 if score is None else score
        text = item.get("text", "")
        context_text += (
            f"第{index}块,标题:{title},来源:{source},"
            f"置信度:{score},内容:{text}\n"
        )

    # ---- 加载历史对话 ----
    history_list = history_repository.list_recent(session_id, limit=6)

    if not history_list:
        logger.warning(f"当前会话{session_id}没有对话记录,提前跳出")
        history = "无对话记录"
    else:
        # 是否存在至少一条带 item_names 的有效记录
        final_history_list = [
            it for it in history_list if it.get("item_names")
        ]
        if not final_history_list:
            logger.warning(f"当前会话{session_id}没有有效的对话记录,提前跳出")
            history = "无有效对话记录"
        else:
            # 注意:这里按你原逻辑,遍历的是全量 history_list;
            # 如果只想拼有效记录,把下面的 history_list 换成 final_history_list
            parts = []
            for index, it in enumerate(history_list, start=1):
                if it.get("role") == "user":
                    prefix = "提问:"
                    content = it.get("rewritten_query") or it.get("text") or ""
                else:
                    prefix = "回答:"
                    content = it.get("text") or ""
                names = ",".join(it.get("item_names") or [])
                parts.append(f"序号{index}:{prefix}{content}关联主体:{names}\n")
            history = "".join(parts)

    final_prompt = load_prompt(
        "answer_out",
        question=question,
        context=context_text,
        history=history,
        item_names=item_names,
    )
    return final_prompt
def llm_get_answer(state: QueryGraphState,prompt:str):
    #获取到llm
    model = llm_provider.chat()
    #封装提示词
    messages = [HumanMessage(content=prompt)]
    answer=""
    #判断是不是流模式
    is_stream = state.get("is_stream")
    if is_stream:
        for chunk in model.stream(messages):
            content = chunk.content
            if content:  # 过滤空 chunk
                push_to_session(state["session_id"], SSEEvent.DELTA, {"content": content})
                answer += content
    else:
        answer=model.invoke(messages).content
    state["answer"] = answer
    logger.info(f"答案生成完成,答案为:{answer}")
def get_urls(state: QueryGraphState):
    #这里url有两个来源一个是从文档当中提取的 另一个是reranker_list当中自带的
    """
    从重排文档中提取可用于前端展示的图片链接。

    Args:
        reranked_docs: 重排后的文档列表，可能包含网页直链或 Markdown 图片语法。

    Returns:
        list[str]: 去重后的图片 URL 列表。
    """
    image_urls: list[str] = []
    reranked_docs = state.get("reranked_docs",[])
    reg = re.compile(r"\!\[.*?\]\((.*?)\)")
    for doc in reranked_docs:
        url = doc.get("url")
        text = doc.get("text")
        if url and url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")) and url not in image_urls:
            image_urls.append(url)
        if text:
            for image_url in reg.findall(text):
                if image_url not in image_urls:
                    image_urls.append(image_url)
    logger.info(f"已完成url提取,长度为:{len(image_urls)}")
    return image_urls
def save_answer_to_state(state: QueryGraphState):
    #保存答案到历史记录当中
    history_repository.save_message(
        session_id = state.get("session_id"),
        role = "assistant",
        text = state.get("answer"),
        rewritten_query = state.get("rewritten_query"),
        item_names = state.get("item_names"),
        image_urls = state.get("image_urls"),
    )
def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """
    答案生成服务：
    1. 检查前置答案（如有追问或拒绝回答，直接输出）
    2. 构建 Prompt（用户问题 + 历史对话 + TopK 文档）
    3. 调用 LLM 生成最终答案（支持流式推送）
    4. 从引用文档中提取图片 URL
    5. 写入 MongoDB 历史记录
    6. 回写 answer 和 image_urls
    """
    # 初始化一个空字符串，可能是用于后续处理
    ""
    
    # 关键点：return 必须保留 session_id！
    flag=check_pre_answer(state)
    if not flag:
        #构建提示词
        prompt=load_prompt_func(state)
        #调用llm生成答案  注意分别处理流式和非流式 保存到state当中
        llm_get_answer(state,prompt)
        #提取url
        state["image_urls"]=get_urls(state)
        #保存答案到列表
    save_answer_to_state(state)
    return state


