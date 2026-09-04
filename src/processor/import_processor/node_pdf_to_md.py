"""
    @Desc   :
    @Time   :2026/9/3 15:07
    @Author :爱吃肯德基
"""
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any

import requests
from dotenv import load_dotenv
from requests import Response

from common.logging.logger import step_log
from utils.path_util import PROJECT_ROOT
from common.logging.logger import node_log, logger
from utils.task_utils import add_running_task, add_done_task
from processor.import_processor.state import ImportGraphState, create_default_state

# 提前加载.env配置文件（必须在读取环境变量前执行，确保os.getenv能获取到值）
# 若.env不在项目根目录，可指定路径：load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()


# 定义MinerU服务配置
@dataclass
class MineruConfig:
    base_url: str
    api_token: str


mineru_config = MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL"),
    api_token=os.getenv("MINERU_API_TOKEN")
)


@node_log('node_pdf_to_md')
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点：pdf转md节点
    为什么叫这个名字：
    核心任务： 将 PDF 非结构化数据转换为 Markdown 结构化数据。
    实现功能：
        1. 日志和任务状态
        2. step_1_validate_paths路径校验
        3. step_2_upload_and_poll MinerU 的交互
        4. step_3_download_and_extract 下载和解压
        5. 日志和任务状态 return state
    """
    # 记录开始
    add_running_task(state['task_id'], 'node_pdf_to_md')
    # step1 校验路径
    pdf_path, output_file_path = step_1_validate_paths(state)
    # step2 MinerU交互，上传和轮询
    zip_url = step_2_upload_and_poll(pdf_path)

    # step3 下载解压
    md_path = step_3_download_and_extract(zip_url, output_file_path, pdf_path.stem)

    state['md_path'] = str(md_path)

    add_done_task(state['task_id'], 'node_pdf_to_md')
    return state


@step_log('step_1_validate_paths')
def step_1_validate_paths(state: ImportGraphState) -> Tuple[Path, Path]:
    """
        入参: state
        出参: pdf_path_obj [Path]  local_dir_obj [Path]
        步骤:
            1. 通过state获取对应的地址
            2. 进行非空校验(pdf_path -> none -> 结束 | local_dir 给与默认地址)
            3. 将两个参数转成Path (str -> Path )
            4. 判断pdf_path_obj是否有文件,local_dir_path 是否存在文件夹
                没有文件->抛出异常
                没有文件夹 -> 创建文件mkdir
            5. 返回两个路径对象-元组
        """
    output_file_dir = state['output_file_dir']
    pdf_path = state.get('pdf_path')
    if not pdf_path:
        logger.error('pdf_path 的值为空，无法读取文件，抛出异常！')
        raise ValueError('pdf_path 参数值为空，无法读取文件')
    if not output_file_dir:
        logger.warning('没有传入 output_file_dir 地址，给默认值')
        output_file_dir = PROJECT_ROOT / 'output'
        state['output_file_dir'] = str(output_file_dir)
    output_file_dir_path = Path(output_file_dir)
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.is_file():
        # pdf地址不存在，抛出异常
        logger.error(f"pdf_path:{pdf_path_obj},不存在或者不是一个文件!")
        raise ValueError(f"pdf_path:{pdf_path_obj},不存在或者不是一个文件!")
    if not output_file_dir_path.is_dir():
        logger.warning(f"output_file_dir:{output_file_dir_path}不存在，或者不是文件夹,我们主动创建!")
        output_file_dir_path.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, output_file_dir_path


@step_log('step_2_upload_and_poll')
def step_2_upload_and_poll(pdf_path: Path) -> str:
    """
    步骤:
         1. 参数校验 (MinerU -> 检查下miner url和key)
         2. 申请上传地址 (MinerU) [batch_id]
         3. 向执行地址进行上传文件
         4. 轮询获取返回结果(zip_url) [batch_id]
         5. 返回zip_url
    :param pdf_path: pdf 路径对象
    :return: zip_url
    """
    # ---------------- 1. 校验MinerU 参数 -------------------
    if not mineru_config.base_url or not mineru_config.api_token:
        logger.error("MinerU 配置错误,请检查MinerU 配置!")
        raise ValueError('MinerU 配置错误,请检查MinerU 配置!')
    # --------------2. 申请上传地址和批量id -----------------
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {mineru_config.api_token}"
    }
    file_upload_url, batch_id = apply_upload_url_batch_id(pdf_path)

    # 3. ===========向指定地址进行文件上传===========
    upload_file(file_upload_url, pdf_path)

    # 4. =========== 轮询获取服务器的解析响应结果 ===========
    # 轮询就是死循环,终止条件: 1. 拿到结果  2. 失败了 3. 超时
    # 准备请求数据
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout = 600  # 轮询最大超时时间 单位秒  设置依据： 1页pdf 0.5-1秒时间  假设有600页pdf，设置为600秒比较合适
    interval_time = 3  # 设置轮询的间隔时间
    start_time = time.time()  # 当前的时间秒 浮点类型
    while True:
        # 4.1 判断是否超时了
        if time.time() - start_time > timeout:
            logger.error(f"轮询超时,请检查minerU配置!")
            raise TimeoutError(f"轮询超时,请检查minerU配置!")
        # 4.2 进行轮询请求(鲁棒性)
        try:
            poll_response = requests.get(poll_url, headers=header)
        except Exception as e:
            logger.warning(f"请求出现异常!{interval_time}秒后重试!!")
            time.sleep(interval_time)
            continue
        # 4.3 判断网络请求状态码
        http_poll_status_code = poll_response.status_code
        # 不等于200 但是5xx系列给与机会,希望MinerU珍惜和修复
        if http_poll_status_code != 200:
            if 500 <= http_poll_status_code < 600:
                # 如果是服务器报错  给机会继续重试
                logger.warning(f"可有修复的网络异常,状态码为:{http_poll_status_code}")
                time.sleep(interval_time)
                continue
            else:
                logger.error(f"不可修复的网络状态异常,状态码为:{http_poll_status_code}")
                raise RuntimeError(f"不可修复的网络状态异常,状态码为:{http_poll_status_code}")
        # 4.4 判断业务状态码
        poll_response_dict = poll_response.json()
        if poll_response_dict['code'] != 0:
            logger.error(f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}")
            raise RuntimeError(f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}")
        # 4.5 判断具体的转化状态 state
        extract_result = poll_response_dict['data']['extract_result'][0]
        extract_result_state = extract_result['state']
        # 1. done 终结了 获取url地址 2. failed 终结了 失败  3. 其他进行中给机会
        if extract_result_state == 'done':
            extract_result_url = extract_result['full_zip_url']
            if not extract_result_url:
                logger.error(f"已经完成了解析,但是zip地址为空!!")
                raise RuntimeError(f"已经完成了解析,但是zip地址为空!!")
            # 5. 返回压缩地址即可
            logger.info(f">>>获取mineru服务器的解析结果：{extract_result_url}>>>")
            return extract_result_url
        elif extract_result_state == 'failed':
            logger.error(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
            raise RuntimeError(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
        else:
            logger.warning(f"解析正在进行中,状态:{extract_result_state}!")
            time.sleep(interval_time)
            continue


def step_3_download_and_extract(zip_url, output_file_path:Path, stem) -> Path:

    # ------------------- 1. 下载ZIP ----------------
    response = requests.get(zip_url)
    if response.status_code != 200:
        logger.error(f"从{zip_url}下载文件失败!")
        raise RuntimeError(f"从{zip_url}下载文件失败!")
    save_zip_file_path = output_file_path / f"{stem}_result.zip"
    save_zip_file_path.write_bytes(response.content)
    # ------------------- 2. 解压ZIP ----------------
    unzip_path = output_file_path / stem
    # 如果文件已存在，删除重建，防止旧文件干扰
    if unzip_path.is_dir():
        shutil.rmtree(unzip_path)
    unzip_path.mkdir(parents=True)
    shutil.unpack_archive(save_zip_file_path, unzip_path)
    # ------------------- 3. 查找 md 文件 ----------------
    md_file_list = list(unzip_path.rglob('*.md'))
    target_md_obj = md_file_list[0]
    # 重命名
    final_md_path_obj = target_md_obj.rename(target_md_obj.with_name(f"{stem}.md"))
    logger.info(f"文件{save_zip_file_path}解压成功")
    return final_md_path_obj


def upload_file(file_upload_url, pdf_path: Path):
    """
    向指定url上传文件
    :param file_upload_url:
    :param pdf_path:
    """
    # Path  read_text writer_text  read_bytes writer_bytes
    pdf_file_data = pdf_path.read_bytes()
    # 获取session对象
    with requests.Session() as session:
        session.trust_env = False  # 纯净版的请求头,不随意携带代理的参数
        # 获取session对象
        upload_response = session.put(file_upload_url, data=pdf_file_data)
        # 网络状态
        if upload_response.status_code != 200:
            logger.error(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")
            raise RuntimeError(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")

    logger.info(f">>>向{file_upload_url}上传{pdf_path}成功>>>")


def apply_upload_url_batch_id(pdf_path: Path) -> Tuple[Any, Any]:
    """
    申请上传地址和批量id
    :param pdf_path:
    :return:
    """
    token = mineru_config.api_token
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path}", "data_id": f"{pdf_path.stem}"}
        ],
        "model_version": "vlm"
    }

    response = requests.post(url, json=data, headers=header)

    batch_id, file_upload_urls = response_parse(response)

    if not file_upload_urls:
        logger.error(f"没有申请到上传地址")
        raise RuntimeError(f"没有申请到上传地址")
    if not batch_id:
        logger.error(f"没有批量操作标识")
        raise RuntimeError(f"没有批量操作标识")

    file_upload_url = file_upload_urls[0]
    logger.info(f">>>申请地址成功{file_upload_url}>>>")
    return file_upload_url, batch_id


def response_parse(response: Response) -> tuple[Any, Any]:
    """
    解析response，获取batch_id，file_upload_urls
    :param response:
    :return:
    """
    http_status_code = response.status_code
    if http_status_code != 200:
        logger.error(f"申请上传地址失败,返回状态码为:{http_status_code},请检查MinerU配置!")
        raise RuntimeError(f"申请上传地址失败,返回状态码为:{http_status_code},{response=},请检查MinerU配置!")
    response_dict = response.json()
    # 获取接口调用状态码
    response_dict_code = response_dict['code']
    # 获取接口处理信息
    response_dict_msg = response_dict['msg']
    if response_dict_code != 0:
        logger.error(f"申请地址网络状态成功!但是业务失败!错误码:{response_dict_code},失败信息:{response_dict_msg}")
        raise RuntimeError(
            f"申请地址网络状态成功!但是业务失败!错误码:{response_dict_code},失败信息:{response_dict_msg}")
    # 获取文件上传的链接地址
    file_upload_urls = response_dict['data']['file_urls']
    # 获取批量操作的id，用于轮询获取本次上传解析结果
    batch_id = response_dict['data']['batch_id']
    return batch_id, file_upload_urls


if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")