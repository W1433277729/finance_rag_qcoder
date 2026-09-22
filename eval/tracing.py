"""
    @Desc   : 追踪（tracing）预留接口 —— 当前未接 LangFuse，只留接入点
    @Time   : 2026/9/22
    @Author :爱吃肯德基

接入步骤（需要时按此操作，无需改动本文件以外的代码）：
    1) uv add langfuse
    2) 在 .env 配置：LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
    3) 运行评估时加 --trace，本题的「召回 → RRF → 重排 → 生成」链路就会上报到 LangFuse

已接好的位置：
    - 查询流程：run_rag_eval.run_query_flow 通过 query_app.invoke(state, config={'callbacks': ...}) 传入
    - 判官调用：RAGAS 0.4 的 collections 指标 ascore 目前不透传 callbacks，
      若要追踪判官调用，需在 eval/judge.py 的 llm_factory 处按 ragas 版本能力另行接入
"""
from src.common.logging.logger import logger


def build_trace_callbacks(enabled: bool) -> list:
    """
    返回追踪回调列表；未开启或未安装 langfuse 时返回空列表并给出提示。
    :param enabled: 是否开启追踪（对应 --trace）
    """
    if not enabled:
        return []
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.warning('已指定 --trace，但未安装 langfuse：请先 uv add langfuse 并配置 LANGFUSE_* 后重试；'
                       '本次运行将不做追踪')
        return []
    handler = CallbackHandler()
    logger.info('LangFuse 追踪已开启')
    return [handler]
