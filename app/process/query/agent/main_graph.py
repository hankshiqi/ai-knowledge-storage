from langgraph.graph import StateGraph, END

from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger

# 1. 定义状态图对象，并且指定全局的 state
query_graph = StateGraph(QueryGraphState)

query_graph.add_node(node_search_embedding)
query_graph.add_node(node_search_embedding_hyde)
query_graph.add_node(node_rrf)
query_graph.add_node(node_item_name_confirm)
query_graph.add_node(node_web_search_mcp)
query_graph.add_node(node_rerank)
query_graph.add_node(node_answer_output)

def after_item_name_confirm_route(state:QueryGraphState):
    #判断是否有answer 如果没有说明item_name明确，如果有则直接跳到node_answer_output
    if state['answer']:
        logger.warning(f"node_item_name_confirm_无法继续向后执行: {state['answer']}")
        return "node_answer_output"
    else:
        return "node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"

query_graph.set_entry_point("node_item_name_confirm") #注意这里是节点名称！！！！！
query_graph.add_conditional_edges(
    "node_item_name_confirm",
    after_item_name_confirm_route,
    {
        "node_answer_output": "node_answer_output",
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_web_search_mcp": "node_web_search_mcp"
    }
)
query_graph.add_edge("node_search_embedding", "node_rrf")
query_graph.add_edge("node_search_embedding_hyde", "node_rrf")
query_graph.add_edge("node_web_search_mcp", "node_rrf")
query_graph.add_edge("node_rrf", "node_rerank")
query_graph.add_edge("node_rerank", "node_answer_output")
query_graph.add_edge("node_rerank", END)
query_app = query_graph.compile()

