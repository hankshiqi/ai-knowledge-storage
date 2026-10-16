import json
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from pathlib import Path

from app.rag.import_.config import CHUNK_MAX_SIZE, CHUNK_OVERLAP, CHUNK_SIZE
from app.shared.runtime.logger import logger, step_log
#4.1参数获取和校验
@step_log('node_document_split')
def node_document_split(state: ImportGraphState) -> tuple[str,str]:
    #获取md_content并且检查是否存在对应内容如果没有就从md_path当中去读取
    md_content = state.get("md_content")
    md_path = state.get("md_path")
    file_title = state.get("file_title")
    if not md_content:
        md_content = Path(md_path).read_text(encoding="utf-8")
        state.set("md_content", md_content)
        if not md_content:
            logger.error(f"md_content为空, md_path: {md_path}")
            raise ValueError(f"md_content为空, md_path: {md_path}")
    #检查file_title
    if not file_title:
        file_title = Path(md_path).stem
        state.set("file_title", file_title)

    #给所有文档统一换行符号
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    return md_content, file_title

@step_log('split_by_titles')
#4.2按照语义进行切分
def split_by_titles(md_content: str, file_title: str) -> list[dict]:
    """
    按标题层级做一级粗切
    返回chunks 列表
    """
    chunks:list[dict]=[]
    current_title=""
    current_lines=[]
    is_code_block = False
    chunk_size=0
    md_content_lines = md_content.split('\n')
    reg = re.compile(r"^\s*#{1,6}\s.+")
    for line in md_content_lines:
        #先去去掉首尾空格，如果为空那么就跳过
        line = line.strip() # 用于判断空行和匹配标题
        if not line:
            continue
    # 后续 append 时，如果不在代码块，建议 append raw_line 以保留缩进
        #判断这行是不是代码块 代码块的前后都会有```
        if line.startswith('```') or line.startswith('~~~'):
            is_code_block =not is_code_block
            current_lines.append(line)
            continue
        #判断没有在代码块并且符合正则也就是这个是标题
        if not is_code_block and reg.match(line):
            if current_title and len(current_lines)>1:
                chunks.append({
                    "content": "\n".join(current_lines),
                    "title": current_title,
                    "file_title": file_title,
                })
                chunk_size+=1

            if not current_title and len(current_lines)>0:
                chunks.append({
                    "content": "\n".join(current_lines),
                    "title": "default",
                    "file_title": file_title,
                })
                chunk_size+=1
            current_title = line
            current_lines = [current_title]
        else:
            current_lines.append(line)
    #保存最后一段
    if current_title and len(current_lines)>1:
        chunks.append({
            "content": "\n".join(current_lines),
            "title": current_title,
            "file_title": file_title,
        })
        chunk_size+=1
    #如果没有标题
    if not current_title and len(current_lines)>0:
        chunks.append({
            "content": "\n".join(current_lines),
            "title": "default",
            "file_title": file_title,
        })
        chunk_size+=1
    logger.info(f"按照标题切分完成, chunk_size: {chunk_size}")
    return chunks
@step_log('_split_long_section')
def _split_long_section(chunk: dict) -> list[dict[str, Any]]:
    """
    对超长文本做二次细切
    """
    title = chunk.get("title", "")
    content = chunk.get("content", "")
    file_title = chunk.get("file_title", "")

    # 1. 安全剥离标题（避免切片切错）
    if title and content.startswith(title + "\n"):
        cr_content = content[len(title) + 1:]
    else:
        cr_content = content

    sub_prefix = title + '\n'
    
    # 2. 计算正文允许的最大长度，额外减去 1 是为了容纳 '\n'
    # 如果标题长到超过了最大限制，给一个最小阈值（比如 100），防止程序崩溃
    max_content_len = max(1, CHUNK_MAX_SIZE - len(title) - 1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_content_len,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
    )

    new_chunk_list = []
    # 3. 迭代切分结果
    for index, split_part in enumerate(splitter.split_text(cr_content), start=1):
        split_text = split_part.strip()
        if not split_text:  # 过滤空白
            continue
            
        new_split_text = sub_prefix + split_text
        new_chunk_list.append({
            "title": f"{title}-{index}",
            "content": new_split_text,
            "file_title": file_title,
            "parent_title": title,
            "part": index
        })
        
    logger.info(f"对超长文本做二次细切完成, new_chunk_list_size: {len(new_chunk_list)}")
    return new_chunk_list
@step_log('_merge_short_sections')
def _merge_short_sections(
   refine_chunk_list
) -> list[dict[str, Any]]:
    merged_chunk_list = []
    base_chunk = None
    for next_chunk in refine_chunk_list:
        if base_chunk is None:
            base_chunk = next_chunk
            continue
        #1.判断是否小于最小限制
        #2.判断父标题是否存在并且这两块是否相同
        is_smaller_than_min = len(next_chunk["content"]) < CHUNK_SIZE
        is_same_parent = next_chunk.get("parent_title") and next_chunk.get("parent_title") == base_chunk.get("parent_title")
        if is_smaller_than_min and is_same_parent:
            #拿到当前块的内容
            next_content = next_chunk["content"][len(next_chunk.get("parent_title"))+1:]
            #判断合并之后会不会过长
            is_not_long = len(base_chunk["content"]) + len(next_content) < CHUNK_MAX_SIZE
            #如果不超过就合并
            if is_not_long:
                base_chunk["content"] += "\n" + next_content
            else:
                #超过了就没办法不能合并
                merged_chunk_list.append(base_chunk)
                base_chunk = next_chunk
        else:
            merged_chunk_list.append(base_chunk)
            base_chunk = next_chunk
    if base_chunk is not None:
        merged_chunk_list.append(base_chunk)
    logger.info(f"对短文本做合并完成, merged_chunk_list_size: {len(merged_chunk_list)}")
    return merged_chunk_list

@step_log('refine_chunks')
def refine_chunks(
    sections: list[dict],
    max_len: int = CHUNK_MAX_SIZE,
    min_len: int = CHUNK_SIZE,
) -> list[dict]:
    """
        【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
        执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvus向量库schema）
        :param sections: 步骤3处理后的章节列表
        :param max_len: 单个Chunk最大字符长度
        :return: 长度适中、低碎片化的最终Chunk列表
        """
    # 边界处理：最大长度无效（空/≤0），直接返回原章节，避免切分异常
    if not max_len or max_len <= 0:
        logger.warning(f"步骤4：Chunk最大长度配置无效（{max_len}），跳过精细化处理")
        return sections

    # 阶段1：切分超长章节 → 所有章节长度控制在max_len内
    refined_split = []
    for sec in sections:
        # 对每个章节执行超长切分，结果平铺加入列表（避免嵌套）
        # extend 的作用就是： 把另一个列表（或可迭代对象）里的“元素”，一个个拆出来，直接追加到当前列表的尾部
        refined_split.extend(_split_long_section(sec))
    logger.info(f"步骤4-1：超长章节切分完成，共生成{len(refined_split)}个初始子Chunk")

    # 阶段2：合并过短章节 → 减少碎片化，提升后续检索/大模型调用效果
    final_sections = _merge_short_sections(refined_split)
    logger.info(f"步骤4-2：过短章节合并完成，最终得到{len(final_sections)}个Chunk")

    # 阶段3：父标题兜底 → 适配Milvus向量库schema（parent_title为必填字段）
    # 兜底规则：无parent_title则用自身title，title也无则填空字符串
    for sec in final_sections:
        if not isinstance(sec, dict):
            continue

        # 补全缺失的part字段（默认0），适配Milvus schema
        if "part" not in sec:
            sec["part"] = 0

        if not sec.get("parent_title"):
            sec["parent_title"] = sec.get("title") or ""
    logger.debug(f"步骤4-3：父标题兜底完成，所有Chunk均包含parent_title字段")

    return final_sections

def back_up_chunks_json(refine_chunk_list:list[dict],md_path:str):
    chunk_list_obj = Path(md_path).with_name(f'{Path(md_path).stem}_chunks.json')
    # ✅ 加上 encoding="utf-8" 保平安
    chunk_list_obj.write_text(
        json.dumps(refine_chunk_list, ensure_ascii=False, indent=4), 
        encoding="utf-8"
    )
    logger.info(f"保存refined_chunks到json文件完成,chunk_list_obj: {chunk_list_obj}")
@step_log('split_document')
def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """
    #1.获取参数并且校验
    md_content, file_title = node_document_split(state)
    #2.语义分割，只保留关联标题

    splited_dicts_list=split_by_titles(md_content,file_title)
    refine_chunk_list = refine_chunks(splited_dicts_list, max_len=CHUNK_MAX_SIZE, min_len=CHUNK_SIZE)
    state['chunks']=refine_chunk_list

    back_up_chunks_json(refine_chunk_list,state.get("md_path"))
    return state