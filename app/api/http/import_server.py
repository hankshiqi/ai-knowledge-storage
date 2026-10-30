"""
导入服务 HTTP 入口模块，直接承载导入接口与相关接口业务逻辑。
"""
import shutil
import sys
import uuid
from datetime import datetime
from hashlib import sha256
from mimetypes import guess_type
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from app.api.schema.import_schema import ImportStatusResponse, UploadResponse
from app.infra.config.providers import infra_config
from app import infra
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.process.import_.agent.state import get_default_state
from app.process.import_.agent.main_graph import import_graph
from app.process.import_.agent.state import create_default_state
from app.infra.persistentce.document_repository import document_repository
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PENDING,
    TASK_STATUS_PROCESSING,
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status,
)

app = FastAPI(
    title=infra_config.settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行与状态查询。",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(infra_config.settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
#1返回html文件
@app.get('/html')
def import_html():
    html_path = Path(PROJECT_ROOT) / "app" / "resources" / "html" / "import.html"
    return FileResponse(html_path, media_type=guess_type(html_path.name)[0])
# Keep the graph entry point synchronous so FastAPI runs it in its worker pool.
def import_main_graph(task_id, local_file_path, local_dir, file_hash):
    try:
        #开始状态
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        document_repository.update_status(file_hash, "processing", task_id=task_id)
        state = create_default_state(
            task_id=task_id,
            local_file_path=local_file_path,
            local_dir=local_dir,
            file_hash=file_hash,
        )
        result_state = import_graph.invoke(state)
        document_repository.update_status(
            file_hash,
            "completed",
            task_id=task_id,
            item_name=result_state.get("item_name"),
        )
        #结束状态
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        #异常处理
        update_task_status(task_id, TASK_STATUS_FAILED)
        document_repository.update_status(file_hash, "failed", task_id=task_id, error=str(e))
        logger.error(f"导入任务{task_id}执行失败，错误信息：{e}")

#上传并且处理文件
@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传")
async def upload_file(backgroundtasks:BackgroundTasks,files:list[UploadFile]):
    #获取task_id,local_dir,local_file_path
    task_id = str(uuid.uuid4())
    #获取当前的时间戳
    timestamp = datetime.now().strftime("%Y%m%d")
    local_dir_obj = Path(PROJECT_ROOT / "output" / timestamp / task_id)
    #local_file_path
    upload_file = files[0] #这是存一个封装后的文件对象，要通过read的方式把具体的文件读出来
    original_filename = upload_file.filename or "uploaded_file"
    safe_filename = Path(original_filename).name
    local_file_path_obj = Path(local_dir_obj / safe_filename)
    local_dir_obj.mkdir(parents=True, exist_ok=True)
    #将文件保存到local_file_path_obj当中国
    #如果太大  需要优化
    # local_file_path_obj.write_bytes(upload_file.read())
    add_running_task(task_id,'upload_file')
    file_hash_digest = sha256()
    with local_file_path_obj.open('wb') as file_buffer:
        while chunk := await upload_file.read(1024 * 1024):
            file_buffer.write(chunk)
            file_hash_digest.update(chunk)
    #最后生成整个文件的hash
    file_hash = file_hash_digest.hexdigest()

    reservation = document_repository.reserve(
        file_hash=file_hash,
        file_title=local_file_path_obj.stem,
        task_id=task_id,
        original_filename=original_filename,
        local_file_path=str(local_file_path_obj),
    )
    if not reservation["accepted"]:
        shutil.rmtree(local_dir_obj, ignore_errors=True)
        # This request created no import job. Close its upload step so the
        # temporary task does not remain in the in-memory running list.
        add_done_task(task_id, "upload_file")
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        existing_status = reservation.get("status", "pending")
        if existing_status == "completed":
            task_response_id = task_id
        else:
            task_response_id = reservation.get("task_id") or task_id
        return UploadResponse(
            code=200,
            message=f"{original_filename} 内容已存在，跳过重复导入（{existing_status}）",
            task_ids=[task_response_id],
        )

    add_done_task(task_id,'upload_file')
    update_task_status(task_id, TASK_STATUS_PENDING)
    #调用import_main_graph 异步执行
    backgroundtasks.add_task(
        import_main_graph,
        task_id,
        str(local_file_path_obj),
        str(local_dir_obj),
        file_hash,
    )
    return UploadResponse(
        code=200,
        message=f"{upload_file.filename}上传成功",
        task_ids=[task_id]
    )

#3.状态查询接口
@app.get("/status/{task_id}")
def task_status(task_id: str):
    #获取已经完成的节点列表和正在进行的节点列表
    running_task_list=get_running_task_list(task_id)
    done_task_list=get_done_task_list(task_id)
    #获取处理的状态
    status = get_task_status(task_id)
    return ImportStatusResponse(
        code=200,
        task_id=task_id,
        status=status,
        done_list = done_task_list,
        running_list=running_task_list
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=infra_config.settings.app_host, port=infra_config.settings.import_app_port)
