"""
    @Desc   :
    @Time   :2026/9/3 15:05
    @Author :爱吃肯德基
"""
from pathlib import Path
from src.common.logging.logger import node_log, logger
from src.utils.task_utils import add_running_task, add_done_task
from src.processor.import_processor.state import ImportGraphState


@node_log('node_entry')
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点：入口节点
    为什么叫这个名字：
    实现功能：
        1.接收文件路径
        2.判断文件类型（pdf、md）
        3.设置state中的路由标记（is_md_read_enabled、is_pdf_read_enabled）
    """
    # 记录开始
    add_running_task(state['task_id'], 'node_entry')
    input_file_path: str = state.get('input_file_path')
    if not input_file_path:
        logger.warning('文件路径为空，直接终止当前导入流程')
        return state
    if input_file_path.endswith('.pdf'):
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = input_file_path
    elif input_file_path.endswith('.md'):
        state['is_md_read_enabled'] = True
        state['md_path'] = input_file_path
    else:
        logger.warning(f'节点：resolve_input_file，不支持的文件类型：{input_file_path}，流程终止')
        return state
    state['file_title'] = Path(input_file_path).stem
    # 记录完成
    add_done_task(state['task_id'], 'node_entry')
    return state
