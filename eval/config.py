"""
    @Desc   : RAG 评估配置（题集路径、判官模型、指标开关、断言阈值）
    @Time   : 2026/9/22
    @Author :爱吃肯德基
"""
import os
from pathlib import Path

from src.common.config.lm_config import lm_config

# ---------------------------
# 路径
# ---------------------------
EVAL_DIR = Path(__file__).resolve().parent
DRAFT_DIR = EVAL_DIR / 'dataset' / 'drafts'
DATASET_DIR = EVAL_DIR / 'dataset'
REPORT_DIR = EVAL_DIR / 'reports'
# 冻结后的正式题库（由 build_dataset.py 从 drafts 生成）
DATASET_PATH = DATASET_DIR / 'questions_v1.jsonl'

# ---------------------------
# 判官模型（复用项目 .env 里的 DashScope/百炼 OpenAI 兼容配置）
# ---------------------------
JUDGE_MODEL = os.getenv('EVAL_JUDGE_MODEL') or lm_config.llm_model
JUDGE_BASE_URL = lm_config.base_url
JUDGE_API_KEY = lm_config.api_key
# 判官必须确定性：温度固定 0，否则同一答案多次评分不可比
JUDGE_TEMPERATURE = 0.0
# 判官单次输出上限：faithfulness / factual_correctness 要输出较长的陈述清单，
# 默认上限会被截断（IncompleteOutputException），故显式抬高
JUDGE_MAX_TOKENS = int(os.getenv('EVAL_JUDGE_MAX_TOKENS') or 8192)
# 判官并发（DashScope 有限流，默认保守）
JUDGE_CONCURRENCY = int(os.getenv('EVAL_JUDGE_CONCURRENCY') or 4)

# ---------------------------
# 默认启用的指标
# ---------------------------
# 说明：answer_relevancy / answer_correctness 需要 embeddings，已通过 eval/embeddings.py
# 适配本地 BGE-M3（不产生外部 API 调用）。answer_correctness 与 factual_correctness 语义有重叠，
# 默认不开，需要时用 --metrics 显式指定。
DEFAULT_METRICS = [
    'faithfulness',                 # 忠实度：答案是否只依据检索片段，有无编造
    'answer_relevancy',             # 答案相关性：是否答到了问题上（依赖 embeddings）
    'context_precision',            # 上下文精确率：相关片段是否排在前面（需 ground_truth）
    'context_recall',               # 上下文召回率：该召回的资料是否召回（需 ground_truth）
    'factual_correctness',          # 事实正确性：与参考答案的事实重合度（需 ground_truth）
    'compliance',                   # 合规 rubric（项目自有护栏）
]
# 可选指标：--metrics 里显式加才会跑
OPTIONAL_METRICS = [
    'answer_correctness',           # 与参考答案的事实+语义综合正确性（依赖 embeddings）
]

# ---------------------------
# 断言与阈值
# ---------------------------
# 无资料兜底话术（与 answer_out.prompt 中的固定话术保持一致）
FALLBACK_MARKER = '未检索到足够信息'
# 兜底判定：命中任一标记即认为走了兜底
FALLBACK_MARKERS = [FALLBACK_MARKER, '暂无相关', '没有相关', '未收录']
# 答题耗时告警阈值（秒）
SLOW_QUERY_SECONDS = 60.0
