from mimetypes import guess_type
from pathlib import Path
import sys
import uuid

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from app.api.schema.query import AsyncQueryResponse, ClearHistoryResponse, HistoryItem, HistoryResponse, QueryRequest, QueryResponse
from app.infra.config.providers import infra_config
from app.shared.clients.mongo_history_utils import get_recent_messages
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.infra.config.providers import settings
from app.process.query.agent.main_graph import query_app as query_graph_app
from app.process.query.agent.state import QueryGraphState, create_query_default_state
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
    update_task_status,
)
from app.infra.persistentce.history_repository import history_repository
# 定义fastapi对象
app = FastAPI(
    title=settings.query_app_name,
    description="描述,进行rag查询的服务对象",
    version="0.2.0"
)

# 跨域处理
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*']
)

#返回查询页面
@app.get("/query/html")
def get_html():
    html_path = Path(PROJECT_ROOT) / "app" / "resources" / "html" / "chat.html"
    return FileResponse(html_path, media_type=guess_type(html_path)[0])
#健康查询接口
@app.get("/health")
async def health():
    """服务健康检查"""
    logger.info("健康检查接口调用成功")
    return {"ok": True}
#SSE推送接口
@app.get("/stream/{session_id}")
async def stream_query_result(session_id: str, request: Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
def run_query_graph(session_id, query, is_stream):
    #不管是不是流失都应该清空字典
    clear_task(session_id)
    update_task_status(session_id, TASK_STATUS_PROCESSING,is_stream)
    state = create_query_default_state(session_id = session_id, original_query = query,is_stream = is_stream)   
    try:
        result_state=query_graph_app.invoke(state)
        update_task_status(session_id, TASK_STATUS_COMPLETED,is_stream)
        if is_stream:
            push_to_session(
                    session_id,
                    SSEEvent.FINAL,
                    {
                        "answer":result_state.get('answer') ,
                        "status": "completed",
                        "image_urls": result_state.get('image_urls',[]),
                    }
                )
            logger.info(f"流式输出完成，总长度: {len(result_state.get('answer'))}")


    except Exception as e:
        logger.exception(f"---session_id = {session_id},查询流程出现异常！！{str(e)}")
        update_task_status(session_id, TASK_STATUS_FAILED,is_stream)
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
        

@app.post("/query") #这里query是提交动作
async def query(request:QueryRequest,background_tasks: BackgroundTasks):
    #首先获取session_id,query,以及is_stream
    session_id = request.session_id or uuid.uuid4().hex
    query = request.query 
    is_stream = request.is_stream
    #判断是否是流式输出
    if is_stream:
        #创建sse队列
        create_sse_queue(session_id)
        #启动异步任务
        background_tasks.add_task(run_query_graph, session_id, query, is_stream)
        logger.info(f'已经开启了流式和异步处理')
        return AsyncQueryResponse(
            session_id=session_id,
            message='异步任务进行中.....'
        )
    else:
        #非流式输出
        run_query_graph(session_id, query, is_stream)
        #获取最后一一个节点的answer输出
        answer=get_task_result(session_id,"answer")
        logger.info(f"开启了同步处理,处理结果为{answer}")
        return QueryResponse(
            session_id = session_id,
            answer = answer,
            message = '同步任务处理完成',
            done_list = get_done_task_list(session_id),
        )

@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    delete_count=history_repository.clear_session(session_id)
    logger.info(f"已经删除了{delete_count}条历史记录")
    return ClearHistoryResponse(
        message=f"已经删除了{delete_count}条历史记录",
        deleted_count=delete_count
    )
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    history_list=get_recent_messages(session_id)
    logger.info(f"查询到{len(history_list)}条历史记录")
    return HistoryResponse(
        session_id=session_id,
        items=[
            HistoryItem(
                id=str(item.get('id')),  # 如果 get 不到会变成字符串 'None'，注意检查源头数据
                session_id=item.get('session_id') or "",
                role=item.get('role') or "",
                text=item.get('text') or "",
                rewritten_query=item.get('rewritten_query') or "",
                item_names=item.get('item_names') or [],   # 核心修改：如果是 None，则返回空列表 []
                image_urls=item.get('image_urls') or [],   # 核心修改：解决你刚才的报错
                ts=item.get('ts')
            )
            for item in history_list
        ]
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=infra_config.settings.app_host, port=infra_config.settings.import_app_port)
