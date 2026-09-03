"""
    @Desc   :导入流程的节点
        文档识别(入口节点)    md      -> 图片处理 -> 文本切块 -> 主题意图提取 -> 文本块向量化 -> 数据持久化
                        pdf转 md
    @Time   :2026/9/2 19:19
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
    local_file_path = state.get('local_file_path')
    if not local_file_path:
        logger.warning('文件路径为空，直接终止当前导入流程')
        return state
    if local_file_path.endswith('.pdf'):
        state['is_md_read_enabled'] = True
        state['pdf_path'] = local_file_path
    elif local_file_path.endswith('.md'):
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    else:
        logger.warning(f'节点：resolve_input_file，不支持的文件类型：{local_file_path}，流程终止')
        return state
    state['file_title'] = Path(local_file_path).stem
    # 记录完成
    add_done_task(state['task_id'], 'node_entry')
    return state


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


@node_log('node_md_img')
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点：md文档图片处理节点
    为什么叫这个名字：
    未来实现功能：
    """
    return state


@node_log('node_document_split')
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点：文档切分节点
    为什么叫这个名字：
    未来实现功能：
    """
    return state


@node_log('node_item_name_recognition')
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点：文本块主体识别节点
    为什么叫这个名字：
    未来实现功能：
    """
    return state


@node_log('node_bge_embedding')
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点：文本向量化节点
    为什么叫这个名字：
    未来实现功能：
    """
    return state


@node_log('node_import_milvus')
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点：向量入库节点
    为什么叫这个名字：
    未来实现功能：
    """
    return state
