import asyncio
import json

from agents import mcp_tools_span
from agents.mcp import MCPServerStreamableHttp

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.infra.config.providers import infra_config
def get_data_and_validate(state):
    """
    获取数据并验证
    """
    rewritten_query=state.get("rewritten_query")
    if rewritten_query is None:
        logger.error(f"rewritten_query is None")
        raise ValueError("rewritten_query is None")
    return rewritten_query
async def open_ai_mcp(rewritten_query):
    mcp_server = MCPServerStreamableHttp(
        name="search_mcp",
        client_session_timeout_seconds=300,
        params={
            "url": infra_config.mcp.mcp_base_url,
            "headers": {"Authorization": f"Bearer {infra_config.mcp.api_key}"},
            "timeout": 300,
            "sse_read_timeout": 300,
        },
        cache_tools_list = True,
        max_retry_attempts = 3
    )
    try:
        # 建立MCP服务连接
        await mcp_server.connect()
        # 打印当前可用工具列表，用于调试观测
        tool_list = await mcp_server.list_tools()
        logger.info(f"工具列表:{tool_list}")
        # 调用百炼联网搜索工具，传入检索问句与结果数量
        return await mcp_server.call_tool(
            tool_name="search_pro",
            arguments={"query": rewritten_query},
        )
        #返回的数据格式   "{\"pages\":[{\"snippet\":\"xxx，url\":\"xxx\"}]}\""
    finally:
        # 无论成功失败，最终释放连接资源，避免连接堆积
        await mcp_server.cleanup()
def search_by_web(state: QueryGraphState) -> QueryGraphState:
    """
    网络搜索服务：
    1. 通过 MCP 协议异步调用百炼联网搜索接口
    2. 将用户的查询转化为实时的、结构化的网络搜索结果
    3. 包含标题、链接和摘要
    4. 回写 web_search_docs
    """
    rewritten_query=get_data_and_validate(state)
    mcp_result=asyncio.run(open_ai_mcp(rewritten_query))
    text = mcp_result.content[0].text
    text_dict=json.loads(text)
    return text_dict.get("pages",[])


if __name__ == "__main__":
    test_state = {
        "session_id": "xxxx",
        "is_stream": False,
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
    }
    result_state = search_by_web(test_state)
    print(result_state)