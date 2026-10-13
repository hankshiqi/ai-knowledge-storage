import re

from app.process.import_.agent.state import ImportGraphState

from hashlib import md5
from itertools import zip_longest
from pathlib import Path
import shutil
import time

import requests
from sqlalchemy import exists
from torch import Value
from app import infra
from app.infra.config.providers import infra_config
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log

def validates_and_get_data(state: ImportGraphState) -> tuple[str,Path,Path]:
    """
    验证并获取数据
    """
    #拿到md_path
    md_path = state.get("md_path")
    if not md_path:
        logger.error(f"md_path不存在，业务提前终止")
        raise ValueError("md_path不存在，业务提前终止")
    #拿到对应的path对象检查是否真的有内容
    md_path_obj = Path(md_path)
    if not md_path_obj.exists():
        logger.error(f"md_path内容不存在，业务提前终止")
        raise ValueError("md_path内容不存在，业务提前终止")
    #取出content
    md_content = md_path_obj.read_text(encoding="utf-8")
    if not md_content:
        logger.error(f"md_path内容为空，业务提前终止")
        raise ValueError("md_path内容为空，业务提前终止")
    #将content存到state当中
    state["md_content"] = md_content
    #拿到image_path
    image_path = md_path_obj.parent / "images"
    return md_content,md_path_obj,image_path
def scan_images(image_path_obj: Path, md_content: str) -> list[tuple[str, Path, tuple[str, str]]]:
    image_list = []
    
    # 1. 一次性提取文档里所有的图片引用（兼容 Markdown 和 HTML）
    # 注意：这里不需要带入具体的 image_name，直接提取全局
    all_refs = re.finditer(
        r"!\[.*?\]\((.*?)\)|<img.*?src=[\"'](.*?)[\"']", 
        md_content, 
        re.IGNORECASE
    )
    
    # 2. 构建一个列表，存储 (匹配到的完整字符串, 匹配对象)
    # 为了后续能提取上下文，我们需要保留 match 对象
    ref_matches = []
    for m in all_refs:
        # 提取出引用路径（Markdown 在 group(1)，HTML 在 group(2)）
        url = m.group(1) or m.group(2)
        if url:
            ref_matches.append((url, m))

    # 3. 遍历本地图片
    for image_path_sig in image_path_obj.iterdir():
        image_name = image_path_sig.name
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
            continue

        found_flag = False
        
        # 4. 在提取出的全局引用中查找该图片
        for url, match in ref_matches:
            if image_name in url:
                found_flag = True
                start = match.start()
                end = match.end()
                
                pre_context = md_content[max(0, start - 100):start]
                post_context = md_content[end:min(len(md_content), end + 100)]
                image_list.append((image_name, image_path_sig, (pre_context, post_context)))
                
        if not found_flag:
            logger.warning(f"图片{image_name}在md_content当中没有找到，跳过")

    return image_list


def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """
    md_content,md_path_obj,image_path_obj=validates_and_get_data(state)
    #如果图片文件夹没有文件就提前终止
    if not image_path_obj.is_dir() or not any(image_path_obj.iterdir()):
        logger.info(f"图片文件夹不存在，直接返回state")
        return state
    #进行图片扫描
    image_list = scan_images(image_path_obj,md_content)
    return state