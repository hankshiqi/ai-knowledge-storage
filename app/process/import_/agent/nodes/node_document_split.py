import os

from app.shared.runtime.logger import logger
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.split_service import split_document

@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    """
    add_running_task(state["task_id"], "node_document_split")
    state = split_document(state)
    add_done_task(state["task_id"], "node_document_split")
    return state

if __name__ == '__main__':
    from app.shared.utils.path_util import PROJECT_ROOT
    from app.process.import_.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }
        result_state = node_md_img(test_state)
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"测试成功：最终生成{len(final_chunks)}个有效Chunk")
        
        if not final_chunks:
            logger.warning("⚠️ 警告：chunks 是空的！请检查 split_service.py 内部的过滤逻辑。")
        else:
            logger.info("开始打印 Chunk 详情：")
            for i, chunk in enumerate(final_chunks):
                # 取内容，并截断前100个字符，防止日志太长
                content_preview = chunk.get("content", "")[:100].replace("\n", " ") + "..."
                title = chunk.get("title", "无标题")
                file_title = chunk.get("file_title", "未知文件")
                
                logger.info(f"[{i+1}] 标题: {title} | 文件: {file_title} | 内容预览: {content_preview}")