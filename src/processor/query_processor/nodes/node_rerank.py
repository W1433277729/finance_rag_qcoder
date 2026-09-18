"""
    @Desc   :
    @Time   :2026/9/14 16:50
    @Author :爱吃肯德基
"""

import time
import sys

from src.common.logging.logger import node_log, logger, step_log
from src.processor.query_processor.state import QueryGraphState
from src.utils.lm.reranker_utils import get_reranker_model
from src.utils.task_utils import add_running_task, add_done_task

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5


#

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: QueryGraphState):
    # 1. 获取请求参数
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])
    rewritten_query = state.get("rewritten_query")
    # 2. 非空判断
    if (not rrf_chunks) or (not web_search_docs) or (not rewritten_query):
        logger.error(f"rrf_chunks,web_search_docs,rewritten_query参数可能为空,业务无法继续,提前终止!")
        raise ValueError(f"rrf_chunks,web_search_docs,rewritten_query参数可能为空,业务无法继续,提前终止!")
    return rrf_chunks, web_search_docs, rewritten_query


@step_log("step_2_validate_and_get_data")
def step_2_merge_rrf_and_web(rrf_chunks: list[dict], web_search_docs: list[dict]) -> list[dict]:
    """
         两路数据融合,统一数据结构
        :param rrf_chunks:
        :param web_search_docs:
        :return:
        """
    # 1. 定一个融合集合
    merged_list: list = []
    # 2. 循环rrf路
    for chunk in rrf_chunks:
        merged_list.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "title": chunk.get("title"),
                "text": chunk.get("content"),
                "type": chunk.get("type"),
                "url": ""
            }
        )
    # 3. 循环web_search_docs
    for doc in web_search_docs:
        merged_list.append(
            {
                "chunk_id": "",
                "title": doc.get("title"),
                "text": doc.get("snippet"),
                "type": "web",
                "url": doc.get("url")
            }
        )
    logger.info(f"完成两路{len(rrf_chunks)}:{len(web_search_docs)}数据融合,融合后的数量:{len(merged_list)}")
    return merged_list


@step_log("step_3_create_question_answer_pair")
def step_3_create_question_answer_pair(merged_list: list[dict], rewritten_query: str) -> list[list[str]]:
    """
      问题: rewritten_query
      答案: merged_list -> dict -> text 答案
    :param merged_list:
    :param rewritten_query:
    :return:
    """
    question_answer_pair: list[list] = []
    for chunk in merged_list:
        question_answer_pair.append([rewritten_query, chunk.get('text')])
    return question_answer_pair


@step_log("step_4_merge_rrf_and_web")
def step_4_list_score_and_rank(merged_list: list[dict], question_answer_pair: list[list[str]]):
    reranker_model = get_reranker_model()
    score_list = reranker_model.compute_score(question_answer_pair, normalize=True)
    logger.debug(f"完成了数据打分:{score_list}")
    for score, chunk in zip(score_list, merged_list):
        chunk["score"] = score
    # 排序处理
    logger.debug(f"未排序之前的合并列表:{merged_list}")
    merged_list.sort(key=lambda c: c.get('score', 0.0), reverse=True)
    logger.debug(f"未排序之后的合并列表:{merged_list}")


@step_log("step_5_dynamic_topk")
def step_5_dynamic_topk(merged_list):
    max_topk = min(RERANK_MAX_TOPK, len(merged_list))
    min_topk = RERANK_MIN_TOPK
    # 定义topk
    # 场景1: 一个断崖都没有 top_k = max
    top_k = max_topk

    # 场景3: min > max [正常不会,列表的长度小于min] 直接取max
    if max_topk > min_topk:
        # 循环指针
        for index in range(min_topk - 1, max_topk - 1):
            current = merged_list[index]
            next = merged_list[index + 1]
            abs = current.get("score", 0.0) - next.get("score", 0.0)
            ratio = abs / current.get("score")
            # 比较断崖
            if abs > RERANK_GAP_ABS or ratio > RERANK_GAP_RATIO:
                # 发生断崖了
                top_k = index + 1
                logger.debug(
                    f"下标:{index}位置发生了断崖!当前值:{current.get('score')},下一个值:{next.get('score', 0.0)}")
                break
    reranked_docs = merged_list[:top_k]
    return reranked_docs


@node_log("node_rerank")
def node_rerank(state: QueryGraphState):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    logger.info("---Rerank处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 1、参数获取及校验
    rrf_chunks, web_search_docs, rewritten_query = step_1_validate_and_get_data(state)
    # 2、对齐数据格式 rrf_chunks | web_search_docs
    merged_list = step_2_merge_rrf_and_web(rrf_chunks, web_search_docs)
    # 3. 封装问题+答案的列表  -> [[],[],[]]
    # [[问题,答案],[问题,答案]]
    question_answer_pair: list[list[str]] = step_3_create_question_answer_pair(merged_list, rewritten_query)
    # 4、内容打分和排序
    step_4_list_score_and_rank(merged_list, question_answer_pair)
    # 5、进行动态内容截取
    reranked_docs = step_5_dynamic_topk(merged_list)
    # 6、状态更新
    state["reranked_docs"] = reranked_docs

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍"},
        {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍"},
        {"chunk_id": "local_3", "content": "路漫漫其修远兮，吾将上下而求索", "title": "励志诗句"},
    ]
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
    ]
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False,
    }
    result = node_rerank(mock_state)
    print(result)
