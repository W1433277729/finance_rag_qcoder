"""
    @Desc   :
    @Time   :2026/9/14 16:35
    @Author :爱吃肯德基
"""

import json
from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from src.processor.query_processor.state import create_query_default_state

from src.utils.task_utils import *
from src.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from src.utils.clients.mongo_history_utils import *
from src.processor.query_processor.main_graph import query_app

# 定义fastapi对象
app = FastAPI(title="query service", description="掌柜智库查询服务！")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")


# 返回chat.html页面
@app.get("/chat.html")
def chat():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    chat_html_path = current_dir_parent_path / "page" / "chat.html"
    if not chat_html_path.exists():
        logger.error(f"页面不存在：{chat_html_path}")
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{chat_html_path}！")
    logger.info("成功加载 chat.html 页面")
    return FileResponse(chat_html_path)


# 核心查询处理函数
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    logger.info(f"[{session_id}] 开始执行查询流程，流式模式：{is_stream}")
    try:
        clear_task(session_id)  # 清空上游数据
        if is_stream:
            # 流式 创建一个新的队列
            create_sse_queue(session_id)
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
        state = create_query_default_state(
            session_id=session_id,
            is_stream=is_stream,
            original_query=user_query
        )
        logger.info(f"查询图开始执行,传入参数:\n{json.dumps(state, indent=4, ensure_ascii=False)}")
        state = query_app.invoke(state)
        logger.info(f"查询图执行结束,返回结果:\n{json.dumps(state, indent=4, ensure_ascii=False)}")
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
        if is_stream:
            # close -> EventSource
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": state.get("answer"),
                    "status": "completed",
                    "image_urls": state.get("image_urls", []),
                    "references": state.get("references", [])
                }
            )
            logger.info('已经结束：final push_to_session')
        return state
    except Exception as e:
        update_task_status(session_id, "failed", is_stream)
        logger.exception(f"{user_query}:执行图流程报错!")
        push_to_session(session_id=session_id, event=SSEEvent.ERROR, data={
            "error": str(e)
        })


# 查询接口
@app.post("/query")
def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    is_stream = request.is_stream
    logger.info(f"用户发起提问: {user_query} 关联的session_id:{session_id},流式状态:{is_stream}")
    if is_stream:
        background_tasks.add_task(run_query_graph, session_id=session_id, is_stream=is_stream, user_query=user_query)
        logger.info(f"[{session_id}] 流式任务已提交至后台执行")
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        state = run_query_graph(session_id, user_query, is_stream)
        done_list = get_done_task_list(session_id)

        return {
            "session_id": session_id,
            "message": f"{session_id}已经完成问题解析!",
            "answer": state.get("answer"),
            "done_list": done_list,
            "image_urls": state.get("image_urls", []),
            "references": state.get("references", [])
        }


# SSE 流式推送接口
@app.get("/stream/{session_id}")
def stream(session_id: str, request: Request):
    logger.info(f"{datetime.now()}完成{session_id}对应的流式响应接口!")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream"
    )


# 健康检查
@app.get("/health")
def health():
    """服务健康检查"""
    logger.info(f"{datetime.now()}:完成本次健康状态检查!")
    return {"ok": True}

@app.get("/history/{session_id}")
def history(session_id: str, limit: int = 50):
    """
    查询当前会话历史记录
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        logger.info(f"查询{session_id}的聊天记录,查询到:{len(records)}条!")
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": session_id,
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "image_urls":history.get("image_urls", []),
                "ts": r.get("ts")
            })
        return {
            "session_id": session_id,
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")

@app.delete("/history/{session_id}")
def clear_chat_history(session_id: str):
    deleted_count = clear_history(session_id)
    logger.info(f"已经清空:{session_id}对应的聊天记录,删除的数量为:{deleted_count}")
    return {"message": "History cleared", "deleted_count": deleted_count}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)