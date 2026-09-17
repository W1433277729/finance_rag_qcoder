"""
    @Desc   :
    @Time   :2026/9/14 16:49
    @Author :爱吃肯德基
"""

import time
import sys

from src.processor.query_processor.nodes.node_search_embedding import node_search_embedding
from src.processor.query_processor.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from src.processor.query_processor.state import QueryGraphState
from src.utils.task_utils import add_running_task, add_done_task
from src.common.logging.logger import logger, node_log, step_log


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    """
      获取核心参数并且校验
    :param state:
    :return:
    """
    # 1. 获取参数
    embedding_chunks = state.get("embedding_chunks")
    hyde_embedding_chunks = state.get("hyde_embedding_chunks")
    # 2. 非空校验
    if (not embedding_chunks) or (not hyde_embedding_chunks):
        logger.error(f"embedding_chunks或者hyde_embedding_chunks为空,业务无法继续,提前终止!")
        raise ValueError(f"embedding_chunks或者hyde_embedding_chunks为空,业务无法继续,提前终止!")
    # 3. 返回结果
    return embedding_chunks, hyde_embedding_chunks


@step_log("step_2_use_rrf_rank")
def step_2_use_rrf_rank(data_list: list[tuple[float, list]], k: int = 60):
    """
    使用rrf的权重排名
    :param data_list:
    :param k:
    """
    # chunk_id 及对应的rrf分数
    score_dict: dict[str, float] = {}
    chunk_dict: dict[str, dict] = {}
    for weight, chunks in data_list:
        for rank, chunk in enumerate(chunks, start=1):
            rrf_score = weight * (1 / (k + rank))
            chunk_id = chunk.get('chunk_id')
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + rrf_score
            chunk['score'] = score_dict.get(chunk_id)
            chunk_dict[chunk_id] = chunk  # 同一个id，只保留一个chunk

    # 字典转list
    chunk_list: list[dict] = [chunk for chunk in chunk_dict.values()]
    # 根据 chunk中score进行排名
    logger.debug(f"排序之前,长度:{len(chunk_list)},内容: {chunk_list}")
    chunk_list.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    logger.debug(f"排序之后,长度:{len(chunk_list)},内容: {chunk_list}")

    return chunk_list


@node_log("node_rrf")
def node_rrf(state: QueryGraphState):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web、KG）进行加权融合排序。
    """
    logger.info("---RRF---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 1. 获取并校验参数(state)，返回 embedding_chunks  hyde_embedding_chunks
    embedding_chunks, hyde_embedding_chunks = step_1_validate_and_get_data(state)

    # 2、配置权重，叠加rrf排名逻辑
    data_list: list[tuple[float, list]] = [
        (0.5, embedding_chunks),
        (0.5, hyde_embedding_chunks)
    ]
    rrf_chunks = step_2_use_rrf_rank(data_list)

    # 3. 更新state
    state['rrf_chunks'] = rrf_chunks

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
    }

    emb_res = node_search_embedding(mock_state)
    hyde_res = node_search_embedding_hyde(mock_state)
    mock_state["embedding_chunks"] = emb_res.get("embedding_chunks") or []
    mock_state["hyde_embedding_chunks"] = hyde_res.get("hyde_embedding_chunks") or []

    result = node_rrf(mock_state)
    print('打印state')
    print(result)
