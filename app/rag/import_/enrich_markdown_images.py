import re

from langchain.messages import HumanMessage
from minio.deleteobjects import DeleteObject

from app.process.import_.agent.state import ImportGraphState
    
from hashlib import md5
from itertools import zip_longest
from pathlib import Path
import shutil
import time
from app.infra.llm.providers import llm_provider
import requests
from sqlalchemy import exists
from torch import Value
from app import infra
from app.infra.config.providers import infra_config
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log
from app.shared.runtime.load_prompt import load_prompt
from langchain_core.output_parsers import StrOutputParser
import base64
import mimetypes
#导入minio_gateway对象
from app.infra.object_storage.minio_gateway import minio_gateway
@step_log("validates_and_get_data")
def validates_and_get_data(state: ImportGraphState) -> tuple[str,Path,Path]:
    """
    验证并获取数据
    主要是做md内容的校验拿到对应的内容，并且获取图片存放文件夹
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
@step_log("scan_images")
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
    #这里的m指的是一个正则匹配对象
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
@step_log("summarize_images")
def summarize_images(image_list: list[tuple[str, Path, tuple[str, str]]],stem:str)->dict[str,str]:
    """
    利用视觉模型进行图片意图识别
    """
    image_txt_dict ={}

    #获取视觉模型
    lm_model = llm_provider.vision_chat()
    # 1. 把 chain 放到循环外面，只需要定义一次！
    chain = lm_model | StrOutputParser()

    for image_name, image_path, image_context in image_list:
        # 封装提示词 (修正拼写)
        image_text = load_prompt("image_summary", root_folder=stem, image_content=image_context)
        
        image_path_obj = Path(image_path)
        image_base64_str = base64.b64encode(image_path_obj.read_bytes()).decode()
        
        # 自动获取 MIME 类型并容错
        mime_type, _ = mimetypes.guess_type(str(image_path_obj))
        if not mime_type:
            mime_type = "image/jpeg" # 兜底默认值

        human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": image_text
                },
                {
                    "type": "image_url", # 必须是 image_url
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64_str}"
                    }
                }
            ]
        )
        
        # 执行调用 chain (必须使用 .invoke)
        image_summary = chain.invoke([human_message])
        
        image_txt_dict[image_name] = image_summary
        logger.info(f"图片{image_name}的意图识别结果为：{image_summary}")
    return image_txt_dict
@step_log("upload_images_and_replace")
def upload_images_and_replace(md_content:str,summarize_images:dict[str,str],image_context_list:list[tuple[str, Path, tuple[str, str]]],stem:str)->str:
    """_summary_

    Args:
        md_content (str): _description_
        summarize_images (dict[str,str]): _description_
        image_context_list (list[tuple[str, Path, tuple[str, str]]]): _description_
        stem (str): _description_

    Returns:
        str: _description_
    """
    #获取minio的gateway
    minio_gateway_obj = minio_gateway.client() #注意不要取相同的名字！！！！！并且这里是调用第三方库得到的对象并不是我们定义的那个
    #先去查询当前目录下面的图片
        # 列出当前文档(stem)在MinIO中已存在的所有图片
    object_list = minio_gateway_obj.list_objects(
        bucket_name=minio_gateway.bucket_name,
        #注意这里前面不能带有斜杠，所以从1开始取
        prefix=f"{minio_gateway.image_dir[1:]}/{stem}/",
        recursive=True,
    )
    #再删除
 # 构造批量删除对象列表
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    # 执行批量删除
    #批量删除是“尽力而为”（Best-Effort）的，它不会因为某个文件删除失败就抛出异常并中断整个操作。
    errors = minio_gateway_obj.remove_objects(
        bucket_name=minio_gateway.bucket_name,
        delete_object_list=delete_object_list,
    )
    for error in errors:
        logger.error(f"文件删除状态：{error}")
    #上传文件，每张请求一次，注意健壮性
    image_urls={}
    for image_name, image_path, _ in image_context_list:
        try:
            image_path_obj = Path(image_path)
            object_name = f"{minio_gateway.image_dir.strip('/')}/{stem}/{image_name}"
            #上传文件
            minio_gateway_obj.fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=object_name,
                file_path=image_path_obj,
                content_type=mimetypes.guess_type(image_name)[0],
            )
            #获取文件url
            image_url = minio_gateway.build_image_url(stem, image_name)
            image_urls[image_name]=image_url
            logger.info(f"{image_name}上传成功，上传地址为{image_url}")
        except Exception as e:
            logger.error(f"图片{image_name}上传失败，错误信息为：{e},跳过本次上传")
            continue
    if not image_urls:
        logger.warning(f"图片上传失败，没有上传成功，直接返回md_content")
        return md_content
    #替换md_content中的图片地址
    for image_name, image_url in image_urls.items():
        #取这个图片的摘要
        image_summary = re.sub(
            r"\s+", " ", summarize_images.get(image_name, "")
        ).strip()
        markdown_image = re.compile(r"(!\[[^\]]*\]\()([^)]*)(\))")
        md_content = markdown_image.sub(
            lambda match: f"![{image_summary}]({image_url})"
            if image_name in match.group(2) and image_summary
            else (
                f"{match.group(1)}{image_url}{match.group(3)}"
                if image_name in match.group(2)
                else match.group(0)
            ),
            md_content,
        )
        html_image = re.compile(r"(<img\b[^>]*?\bsrc=[\"'])([^\"']*)([\"'][^>]*>)", re.IGNORECASE)
        md_content = html_image.sub(
            lambda match: f"{match.group(1)}{image_url}{match.group(3)}"
            if image_name in match.group(2) else match.group(0),
            md_content,
        )

    summaries = [
        f"- {image_name}: {summarize_images[image_name]}"
        for image_name in image_urls
        if summarize_images.get(image_name)
    ]
    if summaries:
        md_content += "\n\n## 图片说明\n\n" + "\n".join(summaries)
    return md_content
@step_log("backup_new_md_content")
def backup_new_md_content(md_content_new:str,md_path_obj:Path) -> str:
    """备份新的md_content到md_path_obj的目录下

    Args:
        md_content_new (str): _description_
        md_path_obj (Path): _description_

    Returns:
        Path: _description_
    """
    new_md_path_obj = md_path_obj.parent / f"{md_path_obj.stem}_new.md"
    new_md_path_obj.write_text(md_content_new,encoding="utf-8")
    return str(new_md_path_obj)
@step_log("enrich_markdown_images")
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
    #进行图片扫描,得到图片上下文
    image_list = scan_images(image_path_obj,md_content)
    if not image_list:
        return state
    #利用视觉模型进行图片意图识别
    image_txt_dict=summarize_images(image_list, md_path_obj.stem)
    md_content_new= upload_images_and_replace(
        md_content, image_txt_dict, image_list, md_path_obj.stem
    )
    state["md_content"]=md_content_new
    state["md_path"] = backup_new_md_content(md_content_new,md_path_obj)
    return state
