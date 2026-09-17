"""
    @Desc   :
    @Time   :2026/9/14 16:48
    @Author :爱吃肯德基
"""

import sys

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.common.config.milvus_config import milvus_config
from src.common.logging.logger import node_log, logger, step_log
from src.processor.query_processor.state import QueryGraphState
from src.utils.clients import milvus_utils
from src.utils.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from src.utils.lm.embedding_utils import generate_embeddings

from src.utils.lm.lm_utils import get_llm_client
from src.utils.load_prompt import load_prompt
from src.utils.task_utils import add_done_task, add_running_task


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: QueryGraphState):
    # 1. 获取数据
    item_names: list[str] = state.get("item_names", [])
    rewritten_query: str = state.get("rewritten_query")
    # 2. 参数校验
    if (not item_names) or (not rewritten_query):
        # [ for item in list if xx  else xxx -> 没有]
        # 值  if  xx else 值
        logger.error(f"item_names或者rewritten_query为空,业务无法继续,提前终止!")
        raise ValueError(f"item_names或者rewritten_query为空,业务无法继续,提前终止!")
    # 3. 返回结果
    return item_names, rewritten_query


@step_log("step_2_select_chunks_in_milvus")
def step_2_call_llm_by_rewritten_query(rewritten_query: str) -> str:
    """
    生成假设性回答
    :param rewritten_query:
    :return:
    """
    # 构建模型
    lm_model = get_llm_client()
    # 填充提示词模板
    prompt = load_prompt('hyde_prompt', rewritten_query=rewritten_query)
    message = HumanMessage(content=prompt)
    # 模型调用
    chain = lm_model | StrOutputParser()
    answer = (chain.invoke([message]))
    # 5. 返回结果
    logger.info(f"基于:{rewritten_query},模型给与的假设性回答:{answer}")
    return answer

    pass


@step_log("step_3_select_chunks_in_milvus")
def step_3_select_chunks_in_milvus(item_names: list[str], rewritten_query: str, hyde_answer: str):
    """
        通过假设性回答，会和检索向量数据库
    :param item_names:
    :param rewritten_query:
    :param hyde_answer:
    """
    # 1、假设性回答向量化
    embedding_result = generate_embeddings(["问题:" + rewritten_query + "\n 假设性回答:" + hyde_answer])
    dense_vector = embedding_result.get('dense')[0]
    sparse_vector = embedding_result.get('sparse')[0]

    # 2、构建过滤条件
    item_names_str = ','.join(f'\"{item_name}\"' for item_name in item_names)
    expr: str = f'item_name in [{item_names_str}]'

    # 2、构建混合检索请求
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr,
        limit=5
    )

    # 3、查询向量数据库
    client = get_milvus_client()
    collection_name = milvus_config.chunks_collection
    response = hybrid_search(
        client=client,
        collection_name=collection_name,
        reqs=reqs,
        ranker_weights=(0.5, 0.5),
        norm_score=True,
        limit=5,
        output_fields=['chunk_id', 'item_name', 'file_title', 'title', 'parent_title', 'part', 'content']
    )

    if not response:
        return []
    return response[0]


def step_4_after_deal_milvus_result(real_response: list[dict]) -> list[dict]:
    hyde_embedding_chunks: list[dict] = []
    if real_response:
        for item in real_response:
            chunk = item.get('entity', {})
            hyde_embedding_chunks.append({
                'chunk_id': chunk.get('chunk_id'),
                'item_name': chunk.get('item_name'),
                'file_title': chunk.get('file_title'),
                'title': chunk.get('title'),
                'parent_title': chunk.get('parent_title'),
                'part': chunk.get('part'),
                'content': chunk.get('content'),
                'score': item.get('distance', 0.0),
                'type': 'milvus'
            })
    logger.info(f"完成了问题向量检索!检索的数量:{len(hyde_embedding_chunks)}")
    return hyde_embedding_chunks


@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state: QueryGraphState):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    logger.info("---HyDE 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 1、参数获取及校验，返回item_names， rewritten_query
    item_names, rewritten_query = step_1_validate_and_get_data(state)

    # 2、调用大模型生成假设性答案
    hyde_answer = step_2_call_llm_by_rewritten_query(rewritten_query)

    # 3、通过假设性回答查询向量数据库
    real_response: list[dict] = step_3_select_chunks_in_milvus(item_names, rewritten_query, hyde_answer)

    # 对检索结果扁平化处理
    hyde_embedding_chunks = step_4_after_deal_milvus_result(real_response)

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info("---HyDE 处理结束---")
    return {
        "hyde_embedding_chunks": hyde_embedding_chunks
    }


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False,
    }
    result = node_search_embedding_hyde(mock_state)
    print(result)
