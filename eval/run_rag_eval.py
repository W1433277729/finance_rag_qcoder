"""
    @Desc   : RAG 评估主入口：逐题跑查询流程 → 抽检索/生成产物 → 规则断言 → RAGAS 判官打分 → 出报告
    @Time   : 2026/9/22
    @Author :爱吃肯德基
"""
import argparse
import asyncio
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.logging.logger import logger
from src.processor.query_processor.main_graph import query_app
from src.processor.query_processor.state import create_query_default_state

from eval.config import (
    DATASET_PATH, DEFAULT_METRICS, FALLBACK_MARKERS, JUDGE_CONCURRENCY,
    JUDGE_MODEL, REPORT_DIR, SLOW_QUERY_SECONDS,
)
from eval.judge import build_metrics
from eval.report import build_report
from eval.tracing import build_trace_callbacks

# 检索命中判定：ranker 后的片段里若出现标准来源，即认为该题检索命中
RETRIEVAL_FIELDS = ('source_file', 'item_name', 'product_name')


def file_stem(name: str) -> str:
    """取文件名主干（去目录与扩展名），用于与 file_title 对齐比较。"""
    return Path(str(name or '')).stem


def normalize_for_match(text: str) -> str:
    """
    关键词匹配前的归一化：全角转半角 + 去掉千分位逗号与空白。
    让数字断言与书写格式无关（38048 与 38,048 等价），避免因格式差异误判为缺失。
    """
    normalized = unicodedata.normalize('NFKC', str(text or ''))
    return re.sub(r'[,\s，、]', '', normalized)


# 数字+单位解析：让断言与单位写法无关（38,048百万元 == 380.48亿元）
# value 为换算到基准单位后的数值，dimension 用于避免「元」与「%」互相误匹配
_NUM_UNIT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(亿元|百万元|万元|元|%|‰|倍)?')
_UNIT_SCALE = {
    '元': (1.0, 'money'), '万元': (1e4, 'money'), '百万元': (1e6, 'money'), '亿元': (1e8, 'money'),
    '%': (1.0, 'percent'), '‰': (0.1, 'percent'), '倍': (1.0, 'times'),
}


def _extract_numbers(text: str) -> list[tuple[float, str]]:
    """抽取文本中的（数值, 量纲）列表，数值已换算到基准单位。"""
    numbers: list[tuple[float, str]] = []
    # 先归一化去掉千分位逗号，否则 "60,339.62亿元" 会被截成 "60" 而丢失数值
    for raw, unit in _NUM_UNIT_RE.findall(normalize_for_match(text)):
        scale_and_dim = _UNIT_SCALE.get(unit)
        if not scale_and_dim:
            continue
        scale, dimension = scale_and_dim
        try:
            numbers.append((float(raw) * scale, dimension))
        except ValueError:
            continue
    return numbers


def keyword_hit(answer: str, keyword: str) -> bool:
    """
    关键词是否命中答案：先做归一化子串匹配；若关键词是「数字+单位」，
    再按数值 + 量纲比较，避免同一数值因单位（百万元/亿元）写法不同被判为缺失。
    """
    normalized_keyword = normalize_for_match(keyword)
    if normalized_keyword and normalized_keyword in normalize_for_match(answer):
        return True
    matched = _NUM_UNIT_RE.fullmatch(normalized_keyword)
    if not matched:
        return False
    raw, unit = matched.groups()
    if not unit:
        return False
    scale, dimension = _UNIT_SCALE.get(unit, (None, None))
    if scale is None:
        return False
    target = float(raw) * scale
    for value, ans_dimension in _extract_numbers(answer):
        if ans_dimension != dimension:
            continue
        if abs(value - target) <= 1e-6 * max(1.0, abs(target)):
            return True
    return False


def load_questions(limit: int | None, ids: list[str] | None) -> list[dict]:
    questions = [json.loads(line) for line in DATASET_PATH.read_text(encoding='utf-8').splitlines() if line.strip()]
    if ids:
        wanted = set(ids)
        questions = [q for q in questions if q['id'] in wanted]
    if limit:
        questions = questions[:limit]
    return questions


def matched_gold(doc: dict, gold_sources: set[str]) -> bool:
    """判断一个检索片段是否命中标准来源（按 source_file 主干 / item_name / product_name 匹配）。"""
    if not gold_sources:
        return False
    candidates = {file_stem(doc.get('source_file'))}
    for field in RETRIEVAL_FIELDS[1:]:
        value = doc.get(field)
        if value:
            candidates.add(str(value).strip())
    return bool(candidates & gold_sources)


def retrieve_stats(docs: list[dict], gold_sources: list[str]) -> dict:
    """命中率 / recall / MRR：区分 rerank 后与 rrf 阶段，便于判断分数损失出在哪一步。"""
    gold = set(gold_sources or [])
    if not gold:
        return {'gold_hit': None, 'recall': None, 'mrr': None, 'gold_rank': None}

    hit_rank = None
    hit_sources: set[str] = set()
    for rank, doc in enumerate(docs, start=1):
        for candidate in [file_stem(doc.get('source_file'))] + [
            str(doc.get(f)).strip() for f in RETRIEVAL_FIELDS[1:] if doc.get(f)
        ]:
            if candidate in gold:
                hit_sources.add(candidate)
                hit_rank = hit_rank or rank
    return {
        'gold_hit': hit_rank is not None,
        'recall': round(len(hit_sources) / len(gold), 3),
        'mrr': round(1 / hit_rank, 3) if hit_rank else 0.0,
        'gold_rank': hit_rank,
    }


def run_query_flow(question: dict, run_id: str, callbacks: list | None = None) -> tuple[dict, float]:
    """进程内执行一次查询流程（不走 HTTP、不落 SSE；每题独立 session_id 避免历史串味）。"""
    state = create_query_default_state(
        session_id=f'eval_{run_id}_{question["id"]}',
        original_query=question['question'],
        is_stream=False,
        # 评估时关闭联网搜索：外部来源会污染检索命中与忠实度判定
        disable_web_search=True,
    )
    started = time.time()
    config = {'callbacks': callbacks or [], 'run_name': question['id']}
    final_state = query_app.invoke(state, config=config)
    return final_state, time.time() - started


def classify_branch(final_state: dict) -> str:
    """
    标注该题走的是哪条路径，便于分组分析：
    - with_entity   ：确认到具体金融产品主体
    - knowledge_bypass：无主体但走了全库检索（知识/概念类旁路）
    - no_context    ：检索为空，走无资料兜底
    - clarify       ：confirm 阶段直接反问/拒答，未进入检索
    """
    item_names = final_state.get('item_names') or []
    docs = final_state.get('reranked_docs') or []
    answer = final_state.get('answer') or ''
    if item_names:
        return 'with_entity'
    if docs:
        return 'knowledge_bypass'
    if any(marker in answer for marker in FALLBACK_MARKERS):
        return 'no_context'
    return 'clarify'


# must_not 命中时的否定词：出现即视为「否定式表述」，不算违规
# （例如 must_not 含「保本」时，答案里的「不保本」「非保本浮动收益型」是正确表述，不应判违规）
_NEGATION_CHARS = ('不', '非', '无', '未', '没', '免')
_NEGATION_WINDOW = 3


def keyword_violated(answer: str, keyword: str) -> bool:
    """
    判定答案是否**实质违反** must_not 关键词：命中关键词时再看它前面几个字符，
    若紧邻否定词则视为否定式表述（如「不保本」），不计违规。
    """
    normalized_answer = normalize_for_match(answer)
    normalized_keyword = normalize_for_match(keyword)
    if not normalized_keyword:
        return False
    start = 0
    while True:
        index = normalized_answer.find(normalized_keyword, start)
        if index < 0:
            return False
        prefix = normalized_answer[max(0, index - _NEGATION_WINDOW):index]
        if not any(char in prefix for char in _NEGATION_CHARS):
            return True
        start = index + len(normalized_keyword)


def is_overall_fallback(answer: str) -> bool:
    """
    判定答案是否「整体走了兜底」：兜底话术出现在开头，或答案本身很短（只有那句话）。
    避免把「答完正文、末尾仅在注意事项里补一句无资料提示」误判为整体兜底。
    """
    stripped = (answer or '').strip()
    if any(stripped.startswith(marker) for marker in FALLBACK_MARKERS):
        return True
    if len(stripped) < 250 and any(marker in stripped for marker in FALLBACK_MARKERS):
        return True
    return False


def evaluate_rules(question: dict, final_state: dict, docs: list[dict], refs: list[dict]) -> dict:
    """确定性断言：不花 LLM 成本，先看合规护栏与来源回传是否真的生效。"""
    answer = final_state.get('answer') or ''
    gold = set(question.get('gold_sources') or [])
    # must_have 支持「数字+单位」按数值比较（百万元/亿元等价）
    missing = [k for k in question.get('must_have') or [] if not keyword_hit(answer, k)]
    # must_not 命中后做否定式排除（不保本 / 非保本 等正确表述不计违规）
    violations = [k for k in question.get('must_not') or [] if keyword_violated(answer, k)]
    hit_fallback = is_overall_fallback(answer)
    return {
        'must_have_missing': missing,
        'must_not_violations': violations,
        'fallback_hit': hit_fallback,
        'fallback_expected': bool(question.get('expect_fallback')),
        'fallback_ok': hit_fallback == bool(question.get('expect_fallback')),
        'references_count': len(refs),
        'references_non_empty': len(refs) > 0,
        'references_cover_gold': any(file_stem(r.get('source_file')) in gold for r in refs) if gold else None,
        'references_from_context': all(
            (r.get('source_file'), r.get('document_title')) in
            {(d.get('source_file'), d.get('title')) for d in docs} for r in refs
        ) if refs and docs else None,
        'answer_has_citation_section': '引用来源' in answer,
        'answer_has_risk_section': '风险' in answer,
    }


async def score_with_judge(metrics: dict, record: dict, semaphore: asyncio.Semaphore) -> dict:
    """按题调用 RAGAS 指标（0.4 collections 的 ascore 接口），单指标失败不影响其它指标。"""
    contexts = record['contexts']
    user_input = record['question']
    response = record['answer']
    reference = record.get('ground_truth') or ''

    async def safe(name: str, coro_factory):
        """重试执行一次指标打分；coro_factory 每次调用需返回新的协程（协程对象不能重复 await）。"""
        for attempt in range(2):
            try:
                async with semaphore:
                    result = await coro_factory()
                return {'value': result.value, 'detail': result.to_dict()}
            except Exception as e:
                if attempt == 1:
                    logger.warning(f'[{record["id"]}] 指标 {name} 打分失败：{type(e).__name__}: {e}')
                    return {'value': None, 'error': f'{type(e).__name__}: {e}'}
                await asyncio.sleep(2)

    scores: dict[str, dict] = {}
    if 'faithfulness' in metrics and response and contexts:
        scores['faithfulness'] = await safe('faithfulness', lambda: metrics['faithfulness'].ascore(
            user_input=user_input, response=response, retrieved_contexts=contexts))
    if 'answer_relevancy' in metrics and response:
        scores['answer_relevancy'] = await safe('answer_relevancy', lambda: metrics['answer_relevancy'].ascore(
            user_input=user_input, response=response))
    if reference and contexts:
        if 'context_precision' in metrics:
            scores['context_precision'] = await safe('context_precision', lambda: metrics['context_precision'].ascore(
                user_input=user_input, reference=reference, retrieved_contexts=contexts))
        if 'context_recall' in metrics:
            scores['context_recall'] = await safe('context_recall', lambda: metrics['context_recall'].ascore(
                user_input=user_input, retrieved_contexts=contexts, reference=reference))
        if 'factual_correctness' in metrics:
            scores['factual_correctness'] = await safe('factual_correctness', lambda: metrics['factual_correctness'].ascore(
                response=response, reference=reference))
        if 'answer_correctness' in metrics:
            scores['answer_correctness'] = await safe('answer_correctness', lambda: metrics['answer_correctness'].ascore(
                user_input=user_input, response=response, reference=reference))
    if 'compliance' in metrics and response:
        scores['compliance'] = await safe('compliance', lambda: metrics['compliance'].ascore(
            user_input=user_input, response=response, retrieved_contexts=contexts))
    return scores


async def score_all(records: list[dict], metric_names: list[str]) -> None:
    metrics = build_metrics(metric_names)
    semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)
    tasks = [score_with_judge(metrics, r, semaphore) for r in records]
    for record, scores in zip(records, await asyncio.gather(*tasks)):
        record['scores'] = scores


def main() -> None:
    parser = argparse.ArgumentParser(description='RAG 生成/检索评估')
    parser.add_argument('--limit', type=int, default=None, help='只跑前 N 题（冒烟用）')
    parser.add_argument('--ids', default=None, help='只跑指定题号，逗号分隔，如 kb-01,risk-01')
    parser.add_argument('--metrics', default=','.join(DEFAULT_METRICS),
                        help='判官指标，逗号分隔；传 none 表示只跑规则断言与检索命中')
    parser.add_argument('--tag', default='', help='本次运行的标签，写进报告名')
    parser.add_argument('--trace', action='store_true',
                        help='开启追踪（预留：LangFuse 接入后即可看到每题检索→重排→生成链路）')
    args = parser.parse_args()

    question_ids = [i.strip() for i in args.ids.split(',')] if args.ids else None
    questions = load_questions(args.limit, question_ids)
    metric_names = [] if args.metrics.lower() == 'none' else [m.strip() for m in args.metrics.split(',') if m.strip()]
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + (f'_{args.tag}' if args.tag else '')

    logger.info(f'评估开始：{len(questions)} 题｜指标={metric_names or "仅规则断言"}｜判官={JUDGE_MODEL}｜联网搜索=关闭')
    callbacks = build_trace_callbacks(enabled=args.trace)

    records: list[dict] = []
    for index, question in enumerate(questions, start=1):
        logger.info(f'[{index}/{len(questions)}] {question["id"]} {question["question"]}')
        try:
            final_state, elapsed = run_query_flow(question, run_id, callbacks)
        except Exception as e:
            logger.error(f'[{question["id"]}] 查询流程异常：{type(e).__name__}: {e}', exc_info=True)
            records.append({**question, 'run_error': f'{type(e).__name__}: {e}', 'answer': '', 'contexts': [],
                            'references': [], 'elapsed': 0.0, 'branch': 'error', 'scores': {}})
            continue

        docs = final_state.get('reranked_docs') or []
        rrf_docs = final_state.get('rrf_chunks') or []
        refs = final_state.get('references') or []
        record = {
            **question,
            'answer': final_state.get('answer') or '',
            'contexts': [d.get('text') or '' for d in docs],
            'references': refs,
            'item_names': final_state.get('item_names') or [],
            'rewritten_query': final_state.get('rewritten_query') or '',
            'branch': classify_branch(final_state),
            'elapsed': round(elapsed, 1),
            'docs': [
                {'file_title': file_stem(d.get('source_file')), 'title': d.get('title'),
                 'content_type': d.get('content_type'), 'score': d.get('score')}
                for d in docs
            ],
            'retrieval': retrieve_stats(docs, question.get('gold_sources') or []),
            'retrieval_pre_rerank': retrieve_stats(rrf_docs, question.get('gold_sources') or []),
            'context_count': len(docs),
            'scores': {},
        }
        record['rules'] = evaluate_rules(question, final_state, docs, refs)
        if elapsed > SLOW_QUERY_SECONDS:
            logger.warning(f'[{question["id"]}] 耗时 {elapsed:.1f}s 超过阈值 {SLOW_QUERY_SECONDS}s')
        records.append(record)

    if metric_names:
        logger.info('开始判官打分……')
        asyncio.run(score_all(records, metric_names))
    else:
        logger.info('已跳过判官打分（仅规则断言与检索命中）')

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f'report_{run_id}.json'
    meta = {
        'run_id': run_id,
        'judge_model': JUDGE_MODEL if metric_names else None,
        'metrics': metric_names,
        'questions': len(records),
        'web_search': 'disabled',
        'trace': 'langfuse' if args.trace else 'off',
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    json_path.write_text(json.dumps({'meta': meta, 'records': records}, ensure_ascii=False, indent=2), encoding='utf-8')
    md_path = build_report(meta, records, REPORT_DIR / f'report_{run_id}.md')
    logger.info(f'报告已生成：{json_path} / {md_path}')


if __name__ == '__main__':
    main()
