"""
    @Desc   :
    @Time   :2026/9/14 16:47
    @Author :爱吃肯德基
"""
from langgraph.constants import END
from langgraph.graph import StateGraph

from src.processor.query_processor.nodes.node_answer_output import node_answer_output
from src.processor.query_processor.nodes.node_item_name_confirm import node_item_name_confirm
from src.processor.query_processor.nodes.node_rerank import node_rerank
from src.processor.query_processor.nodes.node_rrf import node_rrf
from src.processor.query_processor.nodes.node_search_embedding import node_search_embedding
from src.processor.query_processor.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from src.processor.query_processor.nodes.node_web_search_mcp import node_web_search_mcp
from src.processor.query_processor.state import QueryGraphState

builder = StateGraph(QueryGraphState)

# 添加节点
builder.add_node('node_item_name_confirm', node_item_name_confirm)
builder.add_node('node_web_search_mcp', node_web_search_mcp)
builder.add_node('node_search_embedding', node_search_embedding)
builder.add_node('node_search_embedding_hyde', node_search_embedding_hyde)
builder.add_node('node_rrf', node_rrf)
builder.add_node('node_rerank', node_rerank)
builder.add_node('node_answer_output', node_answer_output)

# 入口
builder.set_entry_point('node_item_name_confirm')


# 条件路由
def route_after_item_confirm(state: QueryGraphState):
    if state.get('answer'):
        """
        这主要发生在 node_item_name_confirm 节点无法直接确定唯一的商品型号，从而需要“反问用户”或“拒绝回答”的场景。
        具体来说，有以下两种情况会导致 state 中直接出现 answer ，从而跳过后续的检索流程，直接输出：
        1. 多选一（反问用户） ：
        - 场景 ：用户问得太模糊（比如“华为P60”），系统发现数据库里有“华为P60 128G”和“华为P60 Art”两个型号，且置信度都不足以直接确认。
        - 处理 ：节点会生成一条反问句作为 answer ，例如：“您是想问以下哪个产品：华为P60 128G、华为P60 Art？请明确一下型号。”
        - 结果 ：此时不需要再去检索文档了，直接把这句话发给用户让他选。
        2. 查无此人（拒绝回答） ：
        
        - 场景 ：用户问了一个系统里压根没有的商品（比如“小米15”，但库里只有华为的数据），或者评分过低（<0.6）。
        - 处理 ：节点会生成一条拒绝句作为 answer ，例如：“抱歉，未找到相关产品，请提供准确型号以便我为您查询。”
        - 结果 ：同样不需要后续检索，直接结束流程。
        """
        return 'node_answer_output'
    # 直接进入并发搜索
    return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"


# 添加边

builder.add_conditional_edges(
    'node_item_name_confirm',
    route_after_item_confirm,
    {
        "node_answer_output": "node_answer_output",
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_web_search_mcp": "node_web_search_mcp",
    }
)
builder.add_edge('node_web_search_mcp', 'node_rrf')
builder.add_edge('node_search_embedding', 'node_rrf')
builder.add_edge('node_search_embedding_hyde', 'node_rrf')
builder.add_edge('node_rrf', 'node_rerank')
builder.add_edge('node_rerank', 'node_answer_output')
builder.add_edge('node_answer_output', END)

query_app = builder.compile()

if __name__ == '__main__':
    import json
    from src.common.logging.logger import logger
    from src.processor.query_processor.state import create_query_default_state

    logger.info("===== 开始测试 =====")

    initial_state = create_query_default_state(session_id="test_001",
                                               original_query="华夏债券C的风险等级是多少?")
    final_state = None

    # 只输出更最终的状态值（字典形式），不包含节点名称、执行日志、元数据等额外信息  stream_mode默认为updates
    for event in query_app.stream(initial_state):
        for key, value in event.items():
            logger.success(f"节点: {key}")
            final_state = value

    # 格式化输出最终状态
    logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False)}")

    logger.info("图结构:")
    # uv add grandalf
    query_app.get_graph().print_ascii()

    logger.info("===== 测试结束 =====")
