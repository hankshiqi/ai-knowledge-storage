from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from loguru import logger
from dotenv import load_dotenv
load_dotenv() # 记得调用它

from app.process.import_.agent.state import ImportGraphState
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus

#1.给出全局图
import_graph_builder = StateGraph(state_schema=ImportGraphState)
#2.定义路由函数
def router_function(state: ImportGraphState)->str:
    """路由函数

    Args:
        state (ImportGraphState): 传入的全局状态

    Returns:
        str: 返回的节点名称
    """
    file_path = state.get("local_file_path", "未知路径")
    if state.get("is_md_enabled",False): #注意这里第二个参数代表的是默认值
        logger.info(f"文件路径:{state.get(file_path)},文件类型为md,直接转到'node_md_img'节点")
        return "node_md_img"
    elif state.get("is_pdf_enabled",False):
        logger.info(f"文件路径:{state.get(file_path)},文件类型为pdf,转到'node_pdf_to_md'节点")
        return "node_pdf_to_md"
    else:
        logger.warning(f"文件路径:{state.get(file_path)},文件类型不支持,目前仅支持md和pdf文件")
        return END
#2.添加节点
import_graph_builder.add_node("node_entry", node_entry)
import_graph_builder.add_node("node_pdf_to_md", node_pdf_to_md)
import_graph_builder.add_node("node_md_img", node_md_img)
import_graph_builder.add_node("node_document_split", node_document_split)
import_graph_builder.add_node("node_item_name_recognition", node_item_name_recognition)
import_graph_builder.add_node("node_bge_embedding", node_bge_embedding)
import_graph_builder.add_node("node_import_milvus", node_import_milvus)
import_graph_builder.set_entry_point("node_entry")

#3.添加边

import_graph_builder.add_conditional_edges(
    "node_entry",
    router_function,
    {
        "node_pdf_to_md": "node_pdf_to_md",
        "node_md_img": "node_md_img",
        END: END
    }
)
import_graph_builder.add_edge("node_pdf_to_md", "node_md_img")
import_graph_builder.add_edge("node_md_img", "node_document_split")
import_graph_builder.add_edge("node_document_split", "node_item_name_recognition")
import_graph_builder.add_edge("node_item_name_recognition", "node_bge_embedding")
import_graph_builder.add_edge("node_bge_embedding", "node_import_milvus")
import_graph_builder.add_edge("node_import_milvus", END)
#4.编译
import_graph = import_graph_builder.compile()