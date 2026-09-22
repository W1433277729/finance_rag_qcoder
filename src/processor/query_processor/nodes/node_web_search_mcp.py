"""
    @Desc   :
    @Time   :2026/9/14 16:49
    @Author :爱吃肯德基
"""
import asyncio
import json
import sys

from agents.mcp import MCPServerStreamableHttp

from src.common.config.bailian_mcp_config import mcp_config
from src.processor.query_processor.state import QueryGraphState
from src.utils.task_utils import add_done_task, add_running_task
from src.common.logging.logger import logger, node_log, step_log


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.error(f"重写的问题为空,业务无法继续,提前终止!")
        raise ValueError(f"重写的问题为空,业务无法继续,提前终止!")
    return rewritten_query


@step_log("step_2_call_mcp_tool")
async def step_2_call_mcp_tool(rewritten_query: str):
    """
    调用mcp服务，返回查询结果
    :param rewritten_query:
    """
    # 1、创建mcp服务端，连接对应mcp
    mcp_server = MCPServerStreamableHttp(
        name="Streamable HTTP Python Server",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
            "timeout": 10,
        },
        cache_tools_list=True,
        max_retry_attempts=3,
    )

    # 2、主动触发连接
    await mcp_server.connect()
    try:
        # 3、查询工具列表
        tool_list = await mcp_server.list_tools()
        logger.info(f"查询指定服务器:{mcp_config.mcp_base_url},获取工具列表:{tool_list}")
        # 4、调用工具返回结果
        # 这里调用的是阿里云百炼平台的mcp服务，联网搜索mcp
        result = await mcp_server.call_tool(
            tool_name='search_pro',
            arguments={
                "query": rewritten_query
            }
        )
        return result
    except Exception as e:
        logger.exception(f"连接mcp报错,错误信息:{str(e)}")
    finally:
        await mcp_server.cleanup()


@node_log("node_web_search_mcp")
def node_web_search_mcp(state: QueryGraphState):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    logger.info("---node-web-search-mcp处理---")

    # 0、按需关闭联网搜索（评估/离线场景：外部来源会干扰检索指标，直接跳过）
    if state.get("disable_web_search"):
        logger.info("已开启 disable_web_search，跳过联网搜索召回")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
        return {"web_search_docs": []}

    # 1、参数获取及校验，返回 已改写的问题
    rewritten_query: str = step_1_validate_and_get_data(state)
    # 2、创建mcp服务，进行工具调用
    web_search_docs = []
    try:
        result = asyncio.run(step_2_call_mcp_tool(rewritten_query))
        # 3、结果解析（联网失败/无结果时降级为空，不阻断整体查询）
        if result and getattr(result, "content", None):
            text_result: str = result.content[0].text
            web_search_docs = json.loads(text_result).get('pages', [])
    except Exception as e:
        logger.warning(f"联网搜索失败,降级为无联网结果继续查询,原因:{str(e)}")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    logger.info("---node-web-search-mcp处理结束---")
    return {
        "web_search_docs": web_search_docs
    }


if __name__ == "__main__":
    test_state = {
        "session_id": "xxxx",
        "is_stream": False,
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
    }
    result_state = node_web_search_mcp(test_state)
    print(result_state)
