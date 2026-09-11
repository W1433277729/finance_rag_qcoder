"""
    @Desc   :导入流程 State
        文档识别    md      -> 多模态图片理解 -> 文本切块 -> 主题识别意图提取 -> 文本块向量化 -> 数据持久化
                pdf转 md
    @Time   :2026/9/2 18:38
    @Author :爱吃肯德基
"""
import copy
from typing import TypedDict
from src.common.logging.logger import logger


class ImportGraphState(TypedDict):
    # 任务追踪
    task_id: str

    # 流程控制标记
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    # 路径相关
    output_file_dir: str  # 文件夹地址 （pdf/md 输出的文件夹地址）
    input_file_path: str  # 传入文件的地址，不确定pdf、md
    file_title: str  # 文件名
    pdf_path: str  # pdf 地址，pdf文件的地址保存，input_file_path -> pdf_path
    md_path: str  # md 地址，md文件的地址保存, input_file_path -> md_path

    # 内容相关
    md_content: str
    chunks: list
    item_name: str

    # 数据库相关
    embeddings_content: list


# 初始化默认state
graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "output_file_dir": "",
    "input_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖
    :param overrides:
    """
    state: ImportGraphState = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    :return:
    """
    return copy.deepcopy(graph_default_state)


if __name__ == '__main__':
    """
    测试
    """
    # 创建默认状态
    state = create_default_state(input_file_path="万用表RS-12的使用.pdf")
    logger.info(state)
