from itertools import zip_longest
from pathlib import Path
import time

import requests
from torch import Value
from app import infra
from app.infra.config.providers import infra_config
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime import logger

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
        local_dir:Path=PROJECT_ROOT / PDF_PARSE_SERVICE_LOCAL_DIR
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
                return extract_result.get("full_zip_url")
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
    zip_url=upload_pdf_and_poll(pdf_path_obj)
    
    return state