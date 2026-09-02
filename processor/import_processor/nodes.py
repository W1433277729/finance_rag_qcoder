"""
    @Desc   :导入流程的节点
        文档识别(入口节点)    md      -> 图片处理 -> 文本切块 -> 主题意图提取 -> 文本块向量化 -> 数据持久化
                        pdf转 md
    @Time   :2026/9/2 19:19
    @Author :爱吃肯德基
"""
from common.logging.logger import node_log
from processor.import_processor.state import ImportGraphState


@node_log('node_entry')
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点：入口节点
    为什么叫这个名字：
    未来实现功能：
    """
    return state


@node_log('node_pdf_to_md')
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点：pdf转md节点
    为什么叫这个名字：
    未来实现功能：
    """
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
