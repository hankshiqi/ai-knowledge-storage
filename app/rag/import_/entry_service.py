from pathlib import Path

from app.shared.runtime.logger import logger,step_log

from app.process.import_.agent.state import ImportGraphState

#这里指的是对应的实际业务逻辑
@step_log("resolve_input_file")
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    """
    入口识别服务：
    1. 校验 local_file_path
    2. 识别文件类型（PDF / Markdown）
    3. 回写 is_pdf_read_enabled / is_md_read_enabled
    4. 回写 pdf_path / md_path / file_title
    """
    #1.获取local_file_path
    local_file_path = state.get("local_file_path")
    #2.对其进行非空校验，如果为空就记录并且抛出异常
    if not local_file_path:
        logger.error("local_file_path is empty,程序终止")
        raise FileNotFoundError("local_file_path is empty")
    #3.判断是不是md
    if local_file_path.endswith(".md"):
        state['is_md_enabled'] = True
        state['is_pdf_enabled'] = False
        state['md_path'] = local_file_path
    #4.判断是不是pdf
    elif local_file_path.endswith(".pdf"):
        state['is_md_enabled'] = False
        state['is_pdf_enabled'] = True
        state['pdf_path'] = local_file_path
    #5.都不是就做好警告
    else:
        logger.warning("local_file_path is not a pdf or md file")
        state['is_md_enabled'] = False
        state['is_pdf_enabled'] = False
        return state
    #6.获取file_title
    state['file_title'] = Path(local_file_path).stem
    #7.返回处理后的state
    return state


