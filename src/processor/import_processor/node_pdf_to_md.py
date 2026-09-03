"""
    @Desc   :
    @Time   :2026/9/3 15:07
    @Author :爱吃肯德基
"""
from src.common.logging.logger import node_log, logger
from src.utils.task_utils import add_running_task, add_done_task
from src.processor.import_processor.state import ImportGraphState

@node_log('node_pdf_to_md')
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点：pdf转md节点
    为什么叫这个名字：
    未来实现功能：
        1. 调用 MinerU (magic-pdf) 工具。
        2. 将 PDF 转换成 Markdown 格式。
        3. 将结果保存到 state["md_content"]。
    """
    # 记录开始
    add_running_task(state['task_id'], 'node_pdf_to_md')

    return state