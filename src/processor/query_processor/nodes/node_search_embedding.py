"""
    @Desc   :
    @Time   :2026/9/14 16:48
    @Author :爱吃肯德基
"""

import time
import sys
from src.common.logging.logger import logger, node_log
from src.processor.query_processor.state import QueryGraphState
from src.utils.task_utils import add_done_task, add_running_task


@node_log("node_search_embedding")
def node_search_embedding(state: QueryGraphState):
    """
    节点功能：进行向量内容检索
    """
    logger.info("---向量内容检索 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 搜索假设性答案
    print("向量内容检索答案！！")
    time.sleep(1)

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    logger.info("---向量内容检索 处理结束---")
    return {
        "embedding_chunks": "embedding_chunks"
    }
