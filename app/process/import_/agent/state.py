


import copy
import json
from typing import TypedDict

from loguru import logger


class ImportGraphState(TypedDict):

    task_id:str
    #路径相关
    local_file_path:str
    md_path:str
    pdf_path:str
    is_pdf_enabled:bool
    is_md_enabled:bool
    file_title:str
    force_reparse:bool
    #生成文件内容存放路径
    local_dir:str
    md_content:str
    chunks: list
    #主体名称
    item_name:str
    #嵌入后内容
    embedding_content:list

default_state:ImportGraphState = {
    "task_id": "",
    "local_file_path": "",
    "md_path": "",
    "pdf_path": "",
    "is_pdf_enabled": False,
    "is_md_enabled": False,
    "file_title": "",
    "force_reparse": False,
    "local_dir": "",
    "md_content": "",
    "chunks": [],
    "item_name":"",
    "embedding_content":[]
}

def create_default_state(**kargs) -> ImportGraphState:
    """
    创建默认的ImportGraphState对象
    :return: 默认状态字典
    """
    new_state = copy.deepcopy(default_state)
    new_state.update(kargs)
    return new_state

def get_default_state() -> ImportGraphState:
    """
    获取默认的ImportGraphState对象
    :return: 默认状态字典
    """
    return copy.deepcopy(default_state)

if __name__ == "__main__":
    # 测试创建默认状态
    state = create_default_state(task_id="12345", local_file_path="/path/to/file")
    print(state)
    logger.info("本次生成的state{}",json.dumps(state, ensure_ascii=False, indent=4)) #转成json的字符串
