"""
    @Desc   : RAGAS 判官层：判官 LLM 构建 + 指标实例化 + 合规 rubric 加载
    @Time   : 2026/9/22
    @Author :爱吃肯德基
"""
import json

from openai import AsyncOpenAI

from eval import ragas_compat  # noqa: F401  必须先于 ragas 导入（兼容补丁）
from src.common.config.lm_config import lm_config
from src.common.logging.logger import logger
from src.utils.load_prompt import load_prompt
from eval.config import JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MAX_TOKENS, JUDGE_MODEL, JUDGE_TEMPERATURE
from eval.embeddings import get_ragas_embeddings

import ragas.metrics.collections as ragas_metrics
from ragas.llms import llm_factory


def get_judge_llm():
    """构建判官模型：走项目 .env 的 DashScope/百炼 OpenAI 兼容接口，温度固定 0。"""
    if not JUDGE_API_KEY or not JUDGE_BASE_URL:
        raise ValueError('判官模型缺少配置：请在 .env 配置 OPENAI_API_KEY / OPENAI_BASE_URL')
    client = AsyncOpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
    llm = llm_factory(
        JUDGE_MODEL,
        provider='openai',
        client=client,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    logger.info(f'判官模型已就绪：{JUDGE_MODEL}（temperature={JUDGE_TEMPERATURE}, max_tokens={JUDGE_MAX_TOKENS}）')
    return llm


def load_compliance_rubrics() -> dict[str, str]:
    """
    加载合规 rubric（src/common/prompt/compliance_eval.prompt，JSON 格式）。
    注意：RAGAS 的 DomainSpecificRubrics 让判官按这些 rubric 给一个 **1~5** 的整体分，
    不是「每条 rubric 0/1 求和」；报告里按 (v-1)/4 转换到 0~1（见 eval/report.py）。
    """
    raw = load_prompt('compliance_eval')
    rubrics = json.loads(raw)
    return {k: v for k, v in rubrics.items() if not k.startswith('_')}


def build_metrics(metric_names: list[str], llm=None) -> dict[str, object]:
    """
    按名称构建 RAGAS 指标实例（0.4 的 collections 新 API，按题调用 ascore）。
    :param metric_names: 指标名列表，见 eval/config.DEFAULT_METRICS
    :param llm: 判官 LLM，未传则内部构建
    """
    llm = llm or get_judge_llm()
    metrics: dict[str, object] = {}
    # 需要向量相似度的指标才构建 embeddings（本地 BGE-M3，首次调用会加载模型）
    need_embeddings = bool({'answer_relevancy', 'answer_correctness'} & set(metric_names))
    embeddings = get_ragas_embeddings() if need_embeddings else None

    for name in metric_names:
        if name == 'faithfulness':
            metrics[name] = ragas_metrics.Faithfulness(llm=llm)
        elif name == 'context_precision':
            metrics[name] = ragas_metrics.ContextPrecisionWithReference(llm=llm)
        elif name == 'context_recall':
            metrics[name] = ragas_metrics.ContextRecall(llm=llm)
        elif name == 'factual_correctness':
            metrics[name] = ragas_metrics.FactualCorrectness(llm=llm, mode='f1')
        elif name == 'answer_relevancy':
            metrics[name] = ragas_metrics.AnswerRelevancy(llm=llm, embeddings=embeddings)
        elif name == 'answer_correctness':
            metrics[name] = ragas_metrics.AnswerCorrectness(llm=llm, embeddings=embeddings)
        elif name == 'compliance':
            metrics[name] = ragas_metrics.DomainSpecificRubrics(
                llm=llm, rubrics=load_compliance_rubrics(), with_reference=False
            )
        else:
            raise ValueError(f'未知指标：{name}')
    logger.info(f'已构建指标：{list(metrics.keys())}')
    return metrics
