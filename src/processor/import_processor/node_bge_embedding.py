"""
    @Desc   :
    @Time   :2026/9/3 15:11
    @Author :爱吃肯德基
"""
from src.common.logging.logger import node_log, logger
from src.utils.task_utils import add_running_task, add_done_task
from src.processor.import_processor.state import ImportGraphState

@node_log('node_bge_embedding')
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点：文本向量化节点
    为什么叫这个名字：
    未来实现功能：
    """
    return state
