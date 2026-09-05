"""
    @Desc   :
    @Time   :2026/9/3 15:10
    @Author :爱吃肯德基
"""
import base64
import os
import re
from mimetypes import guess_type
from pathlib import Path
from typing import Tuple

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from common.config.lm_config import lm_config
from common.config.minio_config import minio_config
from common.logging.logger import node_log, logger, step_log
from utils.clients.minio_utils import get_minio_client
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.rate_limit_utils import apply_api_rate_limit
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
def step_2_scan_images(md_content: str, image_dir_obj: Path) -> list[Tuple[str, str, Tuple[str, str]]]:
    """
    扫描 MD 文档中的图片，匹配本地图片文件，并截取图片上下文（前后100字符
    :param md_content:
    :param image_dir_obj:
    :return: [(文件名,文件地址,(上文,下文))]
    """
    # 存储最终处理好的图片信息
    image_info_list = []
    # 遍历图片目录
    for image_file_obj in image_dir_obj.iterdir():
        image_name = image_file_obj.name
        image_path = str(image_file_obj)
        # 过滤非图片格式
        if not is_supported_image(image_path):
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


@step_log("step_3_image_summary")
def step_3_image_summary(images_info_list: list[Tuple[str, str, Tuple[str, str]]], root_folder: str) -> dict[str, str]:
    """
    总结图片，生成 {图片名:对应图片描述}
    :param images_info_list:[(文件名,文件地址,(上文,下文))]
    :param root_folder:文件名
    :return:{图片名:对应图片描述}
    """
    summary_img_dict = {}
    # 1.准备模型对象
    vl_model = get_llm_client(model=lm_config.llm_model)
    # 2.循环 images_info_list，获取每张图片信息
    for image_name, image_path, image_content in images_info_list:
        # 将图片信息拼接为提示词
        image_prompt_text = load_prompt('image_summary', root_folder=root_folder, image_content=image_content)
        image_path_obj = Path(image_path)
        img_base64_str = base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")
        message = HumanMessage(
            [
                {
                    "type": "image_url",
                    # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                    "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{img_base64_str}"},
                },
                # 模型支持在以下text字段中传入Prompt，若未传入，则会使用默认的Prompt：Please output only the text content from the image without any additional descriptions or formatting.
                {"type": "text", "text": image_prompt_text},
            ]
        )
        # 调用视觉模型获取结果
        chains = vl_model | StrOutputParser()
        # 添加范围限制
        apply_api_rate_limit()
        # 模型调用
        image_summary = chains.invoke([message])
        # 结果放到字典中
        summary_img_dict[image_name] = image_summary
        logger.info(f"完成:{image_name}的视觉识别,对应的含义:{image_summary}")

    return summary_img_dict


@step_log("step_4_upload_images_get_url")
def step_4_upload_images_get_url(images_info_list: list[Tuple[str, str, Tuple[str, str]]], stem: str):
    """
    上传文件，并获取文件的访问url
    :param images_info_list:
    :param stem:
    """
    # 创建 MinIO client
    minio_client = get_minio_client()
    # 如果存在旧文件，先执行删除操作
    objects_list = minio_client.list_objects(
        bucket_name=minio_config.bucket_name,
        # 去除开头 /
        prefix=minio_config.minio_img_dir[1:] + '/' + stem,
        recursive=True
    )
    delete_object_list = [DeleteObject(obj.object_name) for obj in objects_list]
    # [失败]
    errors = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=delete_object_list
    )
    for error in errors:
        logger.warning(f"删除图片出现问题:{error}")
    # 重新上传本次对应的文件
    image_url_dict: dict[str, str] = {}
    for image_name, image_path, _ in images_info_list:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=minio_config.minio_img_dir + '/' + stem + '/' + image_name,
                file_path=image_path,
                content_type=guess_type(image_name)[0]
            )
            url = f'http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_name}'
            image_url_dict[image_name] = url
            logger.info(f"{image_name}已经完成上传,对应的地址为:{url}")
        except Exception:
            logger.warning(f"{image_name}上传失败,跳过,继续下一张图片传递!!")


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
    images_info_list = step_2_scan_images(md_content, image_dir_obj)

    # =================== step3 调用视觉模型生成图片摘要 ==========================
    summary_img_dict = step_3_image_summary(images_info_list, md_path_obj.stem)

    # =================== step4 将图片信息传到MinIO服务器 ==========================
    step_4_upload_images_get_url(images_info_list, md_path_obj.stem)

    add_done_task(state.get('task_id'), "node_md_img")
    return state


if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from utils.path_util import PROJECT_ROOT

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
