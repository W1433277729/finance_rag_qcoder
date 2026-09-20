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


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: QueryGraphState):
    # 1. 获取数据
    item_names: list[str] = state.get("item_names", [])
    rewritten_query: str = state.get("rewritten_query")

    # 2. 参数校验：rewritten_query 必须存在；item_names 允许为空（知识/概念类查询走全库检索）
    if not rewritten_query:
        logger.error(f"rewritten_query为空,业务无法继续,提前终止!")
        raise ValueError(f"rewritten_query为空,业务无法继续,提前终止!")
    if not item_names:
        logger.info("item_names为空,按知识/概念类问题进行无主体过滤的全库检索")
    return item_names, rewritten_query


@step_log("step_2_select_chunks_in_milvus")
def step_2_select_chunks_in_milvus(item_names: list[str], rewritten_query: str) -> list[dict]:
    # 1、问题向量化，获取稀疏、稠密向量
    result = generate_embeddings([rewritten_query])
    dense_vector = result.get('dense')[0]
    sparse_vector = result.get('sparse')[0]

    # 2、构建过滤条件：有主体则按 item_name 过滤，无主体则不加过滤（全库检索）
    expr: str | None = None
    if item_names:
        item_names_str = ','.join(f'\"{item_name}\"' for item_name in item_names)
        expr = f'item_name in [{item_names_str}]'

    # 3、构建混合检索 AnnSearchRequest
    search_requests = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr,
        limit=5
    )

    # 4、混合检索
    client = milvus_utils.get_milvus_client()
    collection_name = milvus_config.chunks_collection

    response = hybrid_search(
        client=client,
        collection_name=collection_name,
        reqs=search_requests,
        ranker_weights=(0.5, 0.5),
        norm_score=True,
        limit=5,
        output_fields=CHUNK_OUTPUT_FIELDS
    )
    if not response:
        return []
    return response[0]


@step_log("step_3_after_deal_milvus")
def step_3_after_deal_milvus_result(real_response: list[dict]):
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
    # 1、参数校验，返回item_name、rewritten_query
    item_names, rewritten_query = step_1_validate_and_get_data(state)
    # 2、向量数据库进行混合检索
    real_response: list[dict] = step_2_select_chunks_in_milvus(item_names, rewritten_query)

    # 3、查询结果扁平化
    flat_chunks = step_3_after_deal_milvus_result(real_response)

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
