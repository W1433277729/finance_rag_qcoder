"""
    @Desc   :
    @Time   :2026/9/11 18:43
    @Author :爱吃肯德基
"""
import os
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# --------------------------
# 路径兜底：确保「项目根目录」在 sys.path 中
# 本项目内部模块统一使用 src.xxx 绝对导入（例如 src.utils.task_utils）。
# 无论用 `python -m src.api.file_import_service` 还是 IDE 直接运行本文件，
# 都必须保证项目根目录可被 import，否则 src 包无法解析。
# --------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from src.common.logging.logger import logger
from src.processor.import_processor.main_graph import kb_import_app
from src.processor.import_processor.state import get_default_state
from src.utils.path_util import PROJECT_ROOT
from src.utils.task_utils import (
    update_task_status,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
)

# 定义fastapi 对象
app = FastAPI(title='import service', description='掌柜智库查询服务')

# 跨域配置
app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------
# 静态页面路由：返回文件导入前端页面import.html
# 访问地址：http://localhost:8000/import.html
# --------------------------
@app.get('/import.html')
async def get_import_page():
    """
    返回文件导入，前端页面：import.html
    """
    # 拼接HTML文件绝对路径，基于项目根目录定位
    html_abs_path = PROJECT_ROOT/ 'src' / 'page' / 'import.html'
    logger.info(f"前端页面访问，文件绝对路径：{html_abs_path}")

    # 校验文件是否存在，不存在则抛出404异常
    if not os.path.exists(html_abs_path):
        logger.error(f"前端页面文件不存在，路径：{html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")

    # 以FileResponse 返回HTML文件，浏览器自动渲染
    return FileResponse(
        path=html_abs_path,
        media_type='text/html',
    )


# --------------------------
# 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
# --------------------------
def run_graph_task(task_id: str, output_file_dir, input_file_path):
    """
    LangGraph 全流程执行后台任务
    核心流程：初始化状态 -> 流式执行图节点 -> 实时更新任务状态 -> 异常捕获
    :param task_id: 全局唯一任务ID，关联单个文件的全流程
    :param output_file_dir: 该任务的本地存储目录
    :param input_file_path: 上传文件的本地绝对路径
    """
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{input_file_path}")

        # 2. 初始化LangGraph状态：加载默认状态 + 注入当前任务的核心参数
        state = get_default_state()
        state['task_id'] = task_id
        state['input_file_path'] = input_file_path
        state['output_file_dir'] = output_file_dir

        # 3. 流式执行LangGraph全流程（stream模型：实时获取每个节点的执行结果）
        for event in kb_import_app.stream(state, stream_mode='updates'):
            for node_name, node_result in event.items():
                # 记录每个节点完成的日志，包含任务ID和节点名，方便追踪执行顺序
                logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")

        # 4. 全流程执行完成，更新任务全局状态为：已完成
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")
    except Exception as e:
        # 5. 捕获全流程异常，更新全局任务状态为：失败，记录错误日志
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)


@app.post('/upload', summary='文件上传接口', description='支持多文件批量上传，自动触发知识库导入全流程')
async def upload_files(
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(...)
):
    """
    文件上传核心接口
    1. 接受前端上传的多文件（PDF/MD）
    2. 按[日期/任务ID]分层保存到本地输出目录，避免文件冲突
    3. 为每个文件生成唯一 TaskID，启动独立的LangGraph后台处理任务
    4. 实时更新任务状态，供前端轮询监控进度
    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files:
    """
    # 1. 构建本地存储根目录：项目根目录/output/YYYYMMDD
    today_str = datetime.now().strftime('%Y%m%d')
    date_based_root_dir: Path = PROJECT_ROOT / 'output' / today_str
    # 初始化任务ID流标，用于返回给前端（一个文件对应一个TaskID）
    task_ids = []

    # 2. 遍历处理每个上传的文件（多文件批量处理，各自独立生成TaskID）
    for file in files:
        # 生成全局唯一TaskID，使用uuid4，作为单个文件的全流程标识
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")
        # 3. 标记文件上传阶段为运行中，前端轮询可查
        add_running_task(task_id, 'upload_file')
        # 4. 构建该任务的本地独立目录：out/YYYYMMDD/TaskID，避免多文件重名冲突
        task_output_dir: Path = date_based_root_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        # 5. 构建上传文件的本地保存绝对路径
        input_file_path: Path = date_based_root_dir / file.filename

        # 6. 将上传的文件保存到本地临时目录
        with input_file_path.open('wb') as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{input_file_path}")

        # 7. 标记 文件上传阶段 已完成
        add_done_task(task_id, 'upload_file')

        # 8. 将LangGraph全流程处理加入FastAPI后台任务
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(task_output_dir),
            str(input_file_path)
        )
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

    # 9. 所有文件处理完毕，返回上传成功信息和所有TaskID
    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }

@app.get('/status/{task_id}', summary='任务状态查询', description='根据TaskID查询单个文件的处理进度和全局状态')
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info

# --------------------------
# 服务启动入口
# 直接运行此脚本即可启动FastAPI服务，无需额外执行uvicorn命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8000  # 服务端口
    )