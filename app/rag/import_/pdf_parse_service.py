from pathlib import Path
import shutil
import time

import requests

from app.infra.config.providers import infra_config
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import PROJECT_ROOT, logger, step_log
@step_log("validate_pdf_paths")
def validate_pdf_paths(state:ImportGraphState) -> tuple[Path,Path]:
    """
    校验 PDF 路径,并完成文件存在性检验和文件夹的创建
    """ 
    #1.从状态当中获取pdf_path以及local_dir
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")
    #2.校验pdf_path是否存在
    if not pdf_path:
        logger.error("pdf_path is required")
        raise ValueError("pdf_path is required")
    #3.校验local_dir是否存在，不存在则放在默认的路径下面
    if not local_dir:
        logger.warning(f"local_dir为空,放在默认值:根目录/output下")
        local_dir:Path=PROJECT_ROOT / "output"
        state["local_dir"]=str(local_dir)
    #4.将pdf_path以及local_dir转成path
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    #5.判断pdf_path_obj这个pdf对象是不是真实存在
    if not pdf_path_obj.exists():
        logger.error(f"pdf_path:{pdf_path}不存在")
        raise FileNotFoundError(f"pdf_path:{pdf_path}不存在")
    #6.判断local_dir这个文件夹是否真实存在,若不存在创建相应的文件夹
    if not local_dir_obj.is_dir(): 
        logger.warning(f"local_dir:{local_dir}不存在,创建文件夹")
        local_dir_obj.mkdir(parents=True,exist_ok=True)     
    #7.返回对应的path对象
    return (pdf_path_obj,local_dir_obj) 

#向miniU上传文件并且获取解析结果
@step_log("upload_pdf_and_poll")
def upload_pdf_and_poll(pdf_path_obj:Path)->str:
    """
    向miniU上传文件并且获取解析结果
    """
    MINERU_MODEL_VERSION = 'vlm'
    url = f"{infra_config.mineru.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {infra_config.mineru.api_key}"
    }
    data = {
        "files": [
            {"name":f"{pdf_path_obj.name}"}
        ],
        "model_version":MINERU_MODEL_VERSION
    }
    response = requests.post(url, headers=header, json=data)
    if response.status_code != 200:
        logger.error(f"向MINERU申请上传文件失败,网络问题,状态码:{response.status_code}")
        raise RuntimeError(f"向MINERU申请上传文件失败,网络问题,状态码:{response.status_code}")
    response_dict = response.json()
    if response_dict.get("code",-1) != 0 :
        logger.error(f"向MINERU申请上传文件失败,返回码:{response_dict.get('code')}")
        raise RuntimeError(f"向MINERU申请上传文件失败,返回码:{response_dict.get('code')}")
    batch_id = response_dict.get("data",{}).get("batch_id")
    file_urls = response_dict.get("data", {}).get("file_urls") or []
    # 2. 立刻校验，如果是空的，直接报错，不给程序留隐患
    if not file_urls:
        logger.error(f"获取MinerU上传地址失败, 返回数据中 file_urls 为空, 响应信息: {response_dict.get('msg')}")
        raise ValueError("获取MinerU上传地址失败: file_urls 为空")

    # 3. 确认不是空列表后，再安全地取第一个元素
    upload_url = file_urls[0]
    if not batch_id:
        logger.error(f"获取batch_id失败,业务中断")
        raise ValueError(f"获取batch_id失败,业务中断")
    #2.2向指定的url地址发起网络请求并且上传文件，对着upload_url发起put请求
    with requests.Session() as session:
        session.trust_env = False
        upload_response = session.put(url = upload_url,data= pdf_path_obj.read_bytes())
        if upload_response.status_code != 200:
            logger.error(f"向文件服务器上传文件失败,网络问题,状态码:{upload_response.status_code}")
            raise RuntimeError(f"向文件服务器上传文件失败,网络问题,状态码:{upload_response.status_code}")
    #2.3 轮询向minerU获取batch_id解析状态 zip_url
    #法一：等mineru服务器回调api
    #法二：轮询带着batchid，获取解析状态得到zip_url 这里写轮询的方法
    result_url = f"{infra_config.mineru.base_url}/extract-results/batch/{batch_id}"
    start_time = time.time()
    while True:
        #1.如果没有超过最大时间限制那么就继续轮询，否则抛出异常
        if time.time() - start_time > infra_config.mineru.poll_timeout_seconds:
            logger.error(f"轮询超时,超过最大时间限制{infra_config.mineru.poll_timeout_seconds}")
            raise RuntimeError(f"轮询超时,超过最大时间限制{infra_config.mineru.poll_timeout_seconds}")
        #2.网络波动给机会
        try:
            ask_response = requests.get(url = result_url, headers=header)
        except Exception as e:
            logger.warning(f"网络波动,轮询失败,等待重试,错误信息:{e}")
            time.sleep(infra_config.mineru.poll_interval_seconds)
            continue
        #3.网络状态判定，如果为4xx不给机会，如果为5xx就给机会
        if ask_response.status_code !=200:
            if ask_response.status_code >= 500 and ask_response.status_code < 600:
                logger.warning(f"文件服务器问题,等待重试,状态码:{ask_response.status_code}")
                time.sleep(infra_config.mineru.poll_interval_seconds)
                continue
            else:
                logger.error(f"客户端问题,业务中断,状态码:{ask_response.status_code}")
                raise RuntimeError(f"客户端问题,业务中断,状态码:{ask_response.status_code}")
        #4.业务状态判断
        ask_response_dict = ask_response.json()
        if ask_response_dict.get("code",-1) != 0:
            logger.error(f"结果正常返回,但是业务错误,返回码:{ask_response_dict.get('code')}，错误信息:{ask_response_dict.get('msg')}")
            raise RuntimeError(f"结果正常返回,但是业务错误,返回码:{ask_response_dict.get('code')}，错误信息:{ask_response_dict.get('msg')}")
        #有三种处理的状况done表示已经完成可以返回，failed失败抛出异常，running正在解析，converting格式转换中
        extract_results = ask_response_dict.get("data",{}).get("extract_result",[])
        extract_result = None
        if len(extract_results) == 0:
            logger.warning(f"结果正常返回,但是没有解析结果,等待重试")
            time.sleep(infra_config.mineru.poll_interval_seconds)
            continue
        else:
            extract_result = extract_results[0]
            state = extract_result.get("state")
            if state == "done":
                zip_url=extract_result.get("full_zip_url")
                if not zip_url:
                    logger.error(f"解析完成,但是没有zip_url,业务中断")
                    raise RuntimeError(f"解析完成,但是没有zip_url,业务中断")
                return zip_url
            elif state == "failed":
                logger.error(f"解析失败,业务中断,错误信息:{extract_result.get('msg')}")
                raise RuntimeError(f"解析失败,业务中断,错误信息:{extract_result.get('msg')}")
            elif state == "running":
                logger.warning(f"正在解析,等待重试")
                time.sleep(infra_config.mineru.poll_interval_seconds)
                continue
            elif state == "converting":
                logger.warning(f"正在格式转换,等待重试")
                time.sleep(infra_config.mineru.poll_interval_seconds)
                continue
#根据zip_url下载到local_dir当中
    return new_md_path
@step_log("download_and_extract_markdown")
def download_and_extract_markdown(zip_url:str,local_dir_obj:Path,file_name:str)->Path:
    """进行下载解压，重命名，最终返回md_path_obj

    Args:
        zip_url (_type_): _description_
        local_dir_obj (_type_): _description_
        pdf_path_obj_name (_type_): _description_

    Returns:
        Path: _description_
    """
    #1.下载zip
    response= requests.get(url=zip_url,timeout=(3,60))
    if response.status_code != 200:
        logger.error(f"下载zip失败,网络问题,状态码:{response.status_code}")
        raise RuntimeError(f"下载zip失败,网络问题,状态码:{response.status_code}")
    zip_path_obj:Path = local_dir_obj / f"{file_name}.zip"
    zip_path_obj.write_bytes(response.content)
    #2.解压
    #创建解压后的文件夹
    zip_extract_dir = local_dir_obj / f"{file_name}"
    if zip_extract_dir.exists():
        shutil.rmtree(zip_extract_dir) #这里进行递归清理
    zip_extract_dir.mkdir(parents=True,exist_ok=True)
    shutil.unpack_archive(zip_path_obj,zip_extract_dir) #支持所有格式的压缩包
    #递归查询当前文件夹下面的md文件，并且拿到对应的list
    md_path_list = list(zip_extract_dir.rglob("*.md"))
    if len(md_path_list) ==0:
        logger.error(f"下载zip解压后没有md文件,业务中断")
        raise RuntimeError(f"下载zip解压后没有md文件,业务中断")
    md_path_obj=None
    #查是不是等于filename 如果是就可以直接返回
    for sig_md_path in md_path_list:
        if sig_md_path.name == f"{file_name}.md":
            logger.info(f"解压后名称为{file_name}.md,与原来文件名相同，直接返回")
            md_path_obj = sig_md_path
            return md_path_obj
    #查看是否等于full
    for sig_md_path in md_path_list:
        if sig_md_path.name == "full.md":
            md_path_obj = sig_md_path
            break
    if not md_path_obj:
        md_path_obj = md_path_list[0]
    #3.重命名
    """Python 执行这行代码时，它会把 "财务报告.md" 当作一个相对于当前运行目录（D:\Project）的路径。它的实际行为是：

    把 D:\Project\output\财务报告\full.md 剪切。

    粘贴到 D:\Project\财务报告.md。
    所以不能直接rename
    """
    logger.info(f"触发了md的重命名操作，原名称为{md_path_obj}.md，重命名为{file_name}.md")
    new_md_path =md_path_obj.rename(md_path_obj.parent / f"{file_name}.md") #注意这里修改以后会返回一个新的path对象
    return new_md_path
@step_log("parse_pdf_to_markdown")
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
    PDF 解析服务： 
    1. 调用 MinerU
    2. 下载并解压解析结果
    3. 获取 Markdown 路径和正文内容
    4. 回写 md_path / md_content / local_dir
    """
    #1.获取并且校验参数
    pdf_path_obj,local_dir_obj=validate_pdf_paths(state)

    if not state.get("force_reparse", False):
        existing_md_paths = [
            Path(state["md_path"]) if state.get("md_path") else None,
            local_dir_obj / pdf_path_obj.stem / f"{pdf_path_obj.stem}.md",
            local_dir_obj / f"{pdf_path_obj.stem}.md",
        ]
        for existing_md_path in existing_md_paths:
            if (
                existing_md_path is not None
                and existing_md_path.is_file()
                and existing_md_path.stat().st_size > 0
            ):
                state["md_path"] = str(existing_md_path.resolve())
                logger.info(f"发现已有 Markdown，跳过 MinerU 解析并复用：{existing_md_path}")
                return state

    zip_url=upload_pdf_and_poll(pdf_path_obj)
    md_path_obj:Path=download_and_extract_markdown(zip_url,local_dir_obj,pdf_path_obj.stem)
    #更新md_path
    state['md_path'] = str(md_path_obj)
    return state
