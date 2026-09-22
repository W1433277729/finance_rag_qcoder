"""
    @Desc   :
    @Time   :2026/9/14 16:48
    @Author :爱吃肯德基
"""
import sys

from src.common.config.milvus_config import milvus_config
from src.common.logging.logger import logger, node_log, step_log
from src.processor.query_processor.state import QueryGraphState
from src.utils.clients import milvus_utils
from src.utils.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, CHUNK_OUTPUT_FIELDS
from src.utils.lm.embedding_utils import generate_embeddings
from src.utils.task_utils import add_done_task, add_running_task


# 过滤后召回不足该条数时，自动去掉过滤重试一次（避免过滤过狠把召回打成空）
RETRIEVAL_MIN_HITS = 3


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: QueryGraphState):
    # 1. 获取数据
    item_names: list[str] = state.get("item_names", [])
    doc_filters: dict = state.get("doc_filters") or {}
    rewritten_query: str = state.get("rewritten_query")

    # 2. 参数校验：rewritten_query 必须存在；item_names 允许为空
    #    （无主体时可按 doc_filters 收窄到某份资料，或走全库检索）
    if not rewritten_query:
        logger.error(f"rewritten_query为空,业务无法继续,提前终止!")
        raise ValueError(f"rewritten_query为空,业务无法继续,提前终止!")
    if not item_names and not doc_filters:
        logger.info("item_names与doc_filters均为空,按知识/概念类问题进行无过滤的全库检索")
    return item_names, doc_filters, rewritten_query


@step_log("step_2_build_filter_expr")
def step_2_build_filter_expr(item_names: list[str], doc_filters: dict) -> tuple[str | None, str]:
    """
    构造 Milvus 过滤表达式，优先级：主体 > 资料名(file_title) > 不附加过滤。
    互斥不叠加，避免多条件同时生效把召回打成空。
    :return: (表达式或None, 供日志展示的条件描述)
    """
    if item_names:
        names = ','.join(f'"{name}"' for name in item_names)
        return f'item_name in [{names}]', f'item_name in {item_names}'
    file_titles = [str(title).replace('"', '') for title in (doc_filters.get('file_titles') or [])]
    if file_titles:
        titles = ','.join(f'"{title}"' for title in file_titles)
        return f'file_title in [{titles}]', f'file_title in {file_titles}'
    return None, '不附加过滤(全库检索)'


def _search_chunks_with_expr(dense_vector, sparse_vector, expr: str | None) -> list[dict]:
    """用指定过滤条件跑一次混合检索，返回原始结果列表。"""
    search_requests = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr,
        limit=5
    )
    response = hybrid_search(
        client=milvus_utils.get_milvus_client(),
        collection_name=milvus_config.chunks_collection,
        reqs=search_requests,
        ranker_weights=(0.5, 0.5),
        norm_score=True,
        limit=5,
        output_fields=CHUNK_OUTPUT_FIELDS
    )
    if not response:
        return []
    return response[0]


@step_log("step_3_select_chunks_in_milvus")
def step_3_select_chunks_in_milvus(item_names: list[str], doc_filters: dict, rewritten_query: str) -> list[dict]:
    # 1、问题向量化，获取稀疏、稠密向量（只算一次，重试时不重复编码）
    result = generate_embeddings([rewritten_query])
    dense_vector = result.get('dense')[0]
    sparse_vector = result.get('sparse')[0]

    # 2、构造过滤条件
    expr, filter_desc = step_2_build_filter_expr(item_names, doc_filters)

    # 3、混合检索
    chunks = _search_chunks_with_expr(dense_vector, sparse_vector, expr)

    # 4、过滤过狠导致召回不足时，去掉过滤改为全库检索重试一次
    if expr and len(chunks) < RETRIEVAL_MIN_HITS:
        logger.warning(f"过滤条件[{filter_desc}]下仅召回{len(chunks)}条(少于{RETRIEVAL_MIN_HITS}条),"
                       f"去掉过滤改为全库检索重试")
        chunks = _search_chunks_with_expr(dense_vector, sparse_vector, None)
    else:
        logger.info(f"向量内容检索完成,过滤条件:{filter_desc},召回{len(chunks)}条")
    return chunks


@step_log("step_4_after_deal_milvus")
def step_4_after_deal_milvus_result(real_response: list[dict]):
    """
    milvus 原始结果扁平化
    :param real_response:
    """
    flat_chunks: list[dict] = []
    if real_response:
        for item in real_response:
            chunk = item.get('entity', {})
            flat = {field: chunk.get(field) for field in CHUNK_OUTPUT_FIELDS}
            flat['score'] = item.get('distance', 0.0)
            flat['type'] = 'milvus'
            flat_chunks.append(flat)
    logger.info(f"完成了问题向量检索!检索的数量:{len(flat_chunks)}")
    return flat_chunks


@node_log("node_search_embedding")
def node_search_embedding(state: QueryGraphState):
    """
    节点功能：进行向量内容检索
    """
    logger.info("---向量内容检索 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 1、参数校验，返回 item_names、doc_filters、rewritten_query
    item_names, doc_filters, rewritten_query = step_1_validate_and_get_data(state)
    # 2、向量数据库进行混合检索（含过滤优先级与空召回回退）
    real_response: list[dict] = step_3_select_chunks_in_milvus(item_names, doc_filters, rewritten_query)

    # 3、查询结果扁平化
    flat_chunks = step_4_after_deal_milvus_result(real_response)

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    logger.info("---向量内容检索 处理结束---")
    return {
        "embedding_chunks": flat_chunks
    }


if __name__ == "__main__":
    # 模拟测试数据，注意：虽然函数注解是QueryGraphState，测试直接传dict也可以
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机结构图",  # 模拟改写后的查询
        "item_names": ["HAK 180 烫金机"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")

        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            # 【扁平化业务字段，不再读取entity嵌套】
            print(f"chunk_id: {top1.get('chunk_id')}")
            print(f"score(相似度): {top1.get('score')}")
            print(f"type: {top1.get('type')}")
            print(f"Item Name: {top1.get('item_name')}")
            print(f"file_title: {top1.get('file_title')}")
            print(f"Content Preview: {top1.get('content', '')[:120]}...")
        else:
            print("\n>>> 警告：未检索到任何结果，请检查：")
            print("   1.Milvus集合是否存在对应item_name='HAK 180 烫金机'的数据")
            print("   2.环境变量 CHUNKS_COLLECTION 是否配置正确")
            print("   3.filter expr表达式日志，确认过滤条件是否符合预期")

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)
