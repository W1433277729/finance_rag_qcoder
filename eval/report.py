"""
    @Desc   : 评估报告聚合与渲染：总表 / 分组 / 问题清单 / 检索环节诊断
    @Time   : 2026/9/22
    @Author :爱吃肯德基
"""
from pathlib import Path

# 低分阈值：低于该值视为需要人工复核
LOW_SCORE = 0.6

METRIC_LABELS = {
    'faithfulness': '忠实度',
    'answer_relevancy': '答案相关性',
    'context_precision': '上下文精确率',
    'context_recall': '上下文召回率',
    'factual_correctness': '事实正确性',
    'answer_correctness': '答案正确性',
    'compliance': '合规rubric(0~1)',
}

BRANCH_LABELS = {
    'with_entity': '确认到产品主体',
    'knowledge_bypass': '知识类旁路（无主体）',
    'no_context': '无资料兜底',
    'clarify': '反问/拒答',
    'error': '流程异常',
}


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if isinstance(v, (int, float))]
    return round(sum(values) / len(values), 3) if values else None


def _fmt(value) -> str:
    return '—' if value is None else str(value)


def _metric_value(record: dict, name: str, scale: float = 1.0):
    """取指标分值；合规 rubric 是多条之和的标度，按 rubric 条数归一化到 0~1。"""
    value = (record.get('scores') or {}).get(name, {}).get('value')
    if isinstance(value, (int, float)) and scale and scale != 1.0:
        return round(value / scale, 3)
    return value


def _scales(meta: dict) -> dict:
    count = meta.get('compliance_rubric_count') or 1
    return {'compliance': count}


def _rate(records: list[dict], predicate) -> str:
    """按题统计通过率；无适用题时返回 —"""
    applicable = [r for r in records if predicate(r) is not None]
    if not applicable:
        return '—'
    passed = sum(1 for r in applicable if predicate(r))
    return f'{passed}/{len(applicable)}（{round(passed / len(applicable) * 100)}%）'


def _group_rows(records: list[dict], metric_names: list[str], key: str, labels: dict,
                scales: dict | None = None) -> list[str]:
    scales = scales or {}
    rows = ['| 分组 | 题数 | 检索命中 | 平均 recall | 平均 MRR | ' + ' | '.join(
        METRIC_LABELS.get(m, m) for m in metric_names) + ' |',
        '| --- | --- | --- | --- | --- | ' + ' | '.join('---' for _ in metric_names) + ' |']
    keys = sorted({r.get(key) for r in records}, key=lambda k: str(k))
    for group in keys:
        subset = [r for r in records if r.get(key) == group]
        with_gold = [r for r in subset if r.get('retrieval', {}).get('gold_hit') is not None]
        hit = _rate(subset, lambda r: r.get('retrieval', {}).get('gold_hit')) if with_gold else '—'
        recall = _mean([r['retrieval']['recall'] for r in with_gold])
        mrr = _mean([r['retrieval']['mrr'] for r in with_gold])
        metric_cells = [_fmt(_mean([_metric_value(r, m, scales.get(m, 1.0)) for r in subset]))
                        for m in metric_names]
        rows.append(f'| {labels.get(group, group)} | {len(subset)} | {hit} | {_fmt(recall)} | {_fmt(mrr)} | '
                    + ' | '.join(metric_cells) + ' |')
    return rows


def _collect_issues(records: list[dict], metric_names: list[str], scales: dict | None = None) -> list[str]:
    """列出需要人工看的题：断言失败、检索未命中、指标低分或异常。"""
    scales = scales or {}
    lines: list[str] = []
    for r in records:
        reasons: list[str] = []
        rules = r.get('rules') or {}
        if r.get('run_error'):
            reasons.append(f'流程异常：{r["run_error"]}')
        if rules.get('must_not_violations'):
            reasons.append(f'命中违规词：{"、".join(rules["must_not_violations"])}')
        if rules.get('must_have_missing'):
            reasons.append(f'缺关键词：{"、".join(rules["must_have_missing"])}')
        if rules.get('fallback_ok') is False:
            expected = '应走兜底但未兜底' if rules.get('fallback_expected') else '不应兜底却走了兜底'
            reasons.append(expected)
        if rules.get('references_non_empty') is False:
            reasons.append('references 为空')
        if rules.get('references_cover_gold') is False:
            reasons.append('引用来源未覆盖标准来源')
        if r.get('retrieval', {}).get('gold_hit') is False:
            pre = r.get('retrieval_pre_rerank', {}).get('gold_hit')
            reasons.append(f'检索未命中标准来源（RRF 阶段{"命中" if pre else "未命中"}）')
        for name in metric_names:
            score = _metric_value(r, name, scales.get(name, 1.0))
            if score is not None and score < LOW_SCORE:
                reasons.append(f'{METRIC_LABELS.get(name, name)}={score}')
            elif score is None and (r.get('scores') or {}).get(name, {}).get('error'):
                reasons.append(f'{METRIC_LABELS.get(name, name)} 打分失败')

        if reasons:
            answer = (r.get('answer') or '').replace('\n', ' ')[:220]
            lines.append(f'- **{r["id"]}**（{r.get("category")}｜{BRANCH_LABELS.get(r.get("branch"), r.get("branch"))}）：'
                         + '；'.join(reasons))
            lines.append(f'  - 问题：{r.get("question")}')
            lines.append(f'  - 答案节选：{answer}')
    return lines or ['（无：所有题均未触发复核条件）']


def build_report(meta: dict, records: list[dict], out_path: Path) -> Path:
    metric_names = meta.get('metrics') or []
    scales = _scales(meta)
    lines: list[str] = [f'# RAG 评估报告 {meta["run_id"]}', '']
    lines.append('| 项 | 值 |')
    lines.append('| --- | --- |')
    lines.append(f'| 运行时间 | {meta["created_at"]} |')
    lines.append(f'| 题量 | {meta["questions"]} |')
    lines.append(f'| 判官模型 | {_fmt(meta.get("judge_model"))}（temperature=0） |')
    lines.append(f'| 判官指标 | {"、".join(METRIC_LABELS.get(m, m) for m in metric_names) or "仅规则断言"} |')
    lines.append(f'| 联网搜索 | {meta["web_search"]} |')
    lines.append('')

    lines.append('## 1. 总体指标')
    lines.append('')
    lines.append('**检索环节**（仅有标准来源的题参与统计）')
    lines.append('')
    lines.append('| 指标 | 值 |')
    lines.append('| --- | --- |')
    with_gold = [r for r in records if r.get('retrieval', {}).get('gold_hit') is not None]
    hit_rate_after = _rate(with_gold, lambda r: r['retrieval']['gold_hit']) if with_gold else '—'
    lines.append(f'| 标准来源命中率(rank后) | {hit_rate_after} |')
    pre_hit = sum(1 for r in with_gold if r.get('retrieval_pre_rerank', {}).get('gold_hit'))
    pre_rate = f'{pre_hit}/{len(with_gold)}（{round(pre_hit / len(with_gold) * 100)}%）' if with_gold else '—'
    lines.append(f'| 标准来源命中率(RRF阶段) | {pre_rate} |')
    lines.append(f'| 平均 recall | {_fmt(_mean([r["retrieval"]["recall"] for r in with_gold]))} |')
    lines.append(f'| 平均 MRR | {_fmt(_mean([r["retrieval"]["mrr"] for r in with_gold]))} |')
    lines.append('')
    lines.append('**生成与合规**')
    lines.append('')
    lines.append('| 指标 | 值 |')
    lines.append('| --- | --- |')
    for name in metric_names:
        lines.append(f'| {METRIC_LABELS.get(name, name)} 均值 | '
                     f'{_fmt(_mean([_metric_value(r, name, scales.get(name, 1.0)) for r in records]))} |')
    lines.append(f'| 无违规词通过率 | {_rate(records, lambda r: not r.get("rules", {}).get("must_not_violations"))} |')
    lines.append(f'| 关键词齐全通过率 | {_rate(records, lambda r: not r.get("rules", {}).get("must_have_missing"))} |')
    lines.append(f'| 兜底判断正确率 | {_rate(records, lambda r: r.get("rules", {}).get("fallback_ok"))} |')
    lines.append(f'| references 非空率 | {_rate(records, lambda r: r.get("rules", {}).get("references_non_empty"))} |')
    lines.append(f'| 引用覆盖标准来源率 | {_rate(records, lambda r: r.get("rules", {}).get("references_cover_gold"))} |')
    lines.append(f'| 答案含「引用来源」小节率 | {_rate(records, lambda r: r.get("rules", {}).get("answer_has_citation_section"))} |')
    lines.append('')

    lines.append('## 2. 分组对比')
    lines.append('')
    lines.append('### 2.1 按内容类别')
    lines.append('')
    lines.extend(_group_rows(records, metric_names, 'category', {}, scales))
    lines.append('')
    lines.append('### 2.2 按查询路径')
    lines.append('')
    lines.extend(_group_rows(records, metric_names, 'branch', BRANCH_LABELS, scales))
    lines.append('')

    lines.append('## 3. 需要人工复核的题')
    lines.append('')
    lines.append(f'（触发条件：命中违规词 / 缺关键词 / 兜底判断错 / references 异常 / 检索未命中 / 指标 < {LOW_SCORE} / 流程异常）')
    lines.append('')
    lines.extend(_collect_issues(records, metric_names, scales))
    lines.append('')

    lines.append('## 4. 检索环节诊断')
    lines.append('')
    pre_only = [r['id'] for r in with_gold if r.get('retrieval_pre_rerank', {}).get('gold_hit')
                and not r['retrieval']['gold_hit']]
    never = [r['id'] for r in with_gold if not r.get('retrieval_pre_rerank', {}).get('gold_hit')]
    lines.append(f'- RRF 阶段命中但 rerank 后丢失：**{len(pre_only)}** 题 {pre_only or ""}'
                 '（说明问题出在重排序或动态 TopK 截断）')
    lines.append(f'- 两阶段都未命中：**{len(never)}** 题 {never or ""}'
                 '（说明问题出在召回或切分/元数据，需看该题切片是否入库）')
    lines.append('')
    lines.append('## 5. 说明')
    lines.append('')
    lines.append('- `contexts` 取重排序后的最终片段；检索命中按片段元数据（source_file 主干 / item_name / '
                 'product_name）与题目的 gold_sources 匹配，颗粒度到「文件」而非「章节」。')
    lines.append('- 忠实度只衡量「答案是否忠于给定片段」，检索错了它也可能很高分，必须与检索命中率交叉看。')
    lines.append('- `must_have` / `must_not` 是**字面**匹配（数字支持千分位与单位换算）：无法识别否定式表达，'
                 '例如 must_not 含「承诺收益」时，答案里的「不承诺收益」也会被判为违规。'
                 '带否定语义的合规判断请看「合规rubric」的 LLM 评分。')
    lines.append('- 判官打分同一答案存在波动，横向对比请用同一版题库 + 同一判官模型。')

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    return out_path
