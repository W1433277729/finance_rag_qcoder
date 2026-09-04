"""
    @Desc   :
    @Time   :2026/9/3 15:10
    @Author :爱吃肯德基
"""
import re
from pathlib import Path
from typing import Tuple

from common.logging.logger import node_log, logger, step_log
from utils.task_utils import add_running_task, add_done_task
from processor.import_processor.state import ImportGraphState

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]


def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename:
    :return:
    """
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    提取和校验内容，返回图片、md地址
    :param state:
    :return:  md_content 文档内容, md_path_obj 文档地址, image_dir_obj 图片存放文件夹
    """
    md_path = state.get('md_path')
    if not md_path:
        logger.error("md_path变量为空!")
        raise ValueError("md_path变量为空!")
    md_path_obj = Path(md_path)
    # 判断文件是否真的存在
    if not md_path_obj.exists():
        logger.error(f"{md_path_obj}不存在!")
        raise FileNotFoundError(f"{md_path_obj}不存在!")
    # 获取md_content
    md_content = md_path_obj.read_text(encoding="utf-8")
    # 获取图片存放目录
    image_dir_obj = md_path_obj.parent / "images"
    return md_content, md_path_obj, image_dir_obj


@step_log("step_2_scan_images")
def step_2_scan_images(md_content: str, image_dir_obj: Path):
    """
    扫描 MD 文档中的图片，匹配本地图片文件，并截取图片上下文（前后100字符
    :param md_content:
    :param image_dir_obj:
    """
    # 存储最终处理好的图片信息
    image_info_list = []
    # 遍历图片目录
    for image_file_obj in image_dir_obj.iterdir():
        image_name = image_file_obj.name
        image_path = str(image_file_obj)
        # 过滤非图片格式
        if is_supported_image(image_path):
            logger.warning(f'跳过非图片文件{image_name}')
            continue
        # 正则匹配md 文本中的图片语法![](...图片名...)
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        search_match = pattern.search(md_content)
        if not search_match:
            # 没有匹配到
            logger.warning(f'{image_name}没有被md_content引用')
            continue
        # 可以匹配到
        start = search_match.start()
        end = search_match.end()
        pre_context = md_content[max(0, start - 100): start]
        post_context = md_content[end: min(end + 100, len(md_content))]
        logger.debug(
            f"{image_name}在md_content被引用,引用的位置:{start}:{end},截取的上文:{pre_context} , 下文:{post_context}")
        image_info_list.append((image_name, image_path, (pre_context, post_context)))
    logger.info(f"所有图片的上下文信息已经识别完毕,数量为:{len(image_info_list)}")
    return image_info_list


@node_log('node_md_img')
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点：图片处理 (node_md_img) 处理 Markdown 中的图片资源 (Image)。
    """
    add_running_task(state.get('task_id'), "node_md_img")
    # =================== step1 校验参数，获取文档相关信息 =======================
    md_content, md_path_obj, image_dir_obj = step_1_validate_and_get_data(state)
    # 校验是否存在图片
    if (not image_dir_obj.is_dir()) or len(list(image_dir_obj.iterdir())) == 0:
        # 不存在或者是文件 -> true
        # 存在是文件夹  or  没有没有文件 -> false
        logger.info(f"{md_path_obj}对应的md,没有图片内容,无需后续处理,直接跳出!!")
        return state

    # =================== step2 获取图片上下文和信息 =======================
    step_2_scan_images(md_content, image_dir_obj)

    add_done_task(state.get('task_id'), "node_md_img")
    return state
