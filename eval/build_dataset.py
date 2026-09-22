"""
    @Desc   : 题库构建与校验：合并 drafts/*.jsonl → questions_v1.jsonl，并生成人工评审表
    @Time   : 2026/9/22
    @Author :爱吃肯德基
"""
import argparse
import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.config import DATASET_PATH, DRAFT_DIR, DATASET_DIR
from eval.kb_inventory import load_all_chunks

# 类别固定顺序（与 KB content_type 对齐），报告分组时按此顺序
CATEGORY_ORDER = [
    '基金产品资料概要',
    '公司公告摘要',
    '金融术语解释资料',
    '政策解读资料',
    '产品风险揭示书',
    '常见问题FAQ',
]

REQUIRED_FIELDS = {
    'id': str,
    'category': str,
    'question': str,
    'ground_truth': str,
    'gold_sources': list,
    'must_have': list,
    'must_not': list,
    'expect_fallback': bool,
    'expect_compliance': bool,
    'evidence': str,
    'design_note': str,
}


def load_drafts() -> list[dict]:
    """读取 drafts 目录下所有 jsonl 草稿。"""
    questions: list[dict] = []
    for path in sorted(DRAFT_DIR.glob('*.jsonl')):
        lines = path.read_text(encoding='utf-8').splitlines()
        for lineno, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f'{path.name} 第 {lineno} 行不是合法 JSON：{e}') from e
    return questions


def kb_file_titles() -> set[str]:
    """取知识库实际入库的 file_title 集合，作为 gold_sources 的合法取值域。"""
    rows = load_all_chunks()
    return {r.get('file_title') for r in rows if r.get('file_title')}


def normalize_text(text: str) -> str:
    return re.sub(r'[\s，。？?！!、,.]', '', text or '')


def validate(questions: list[dict], valid_titles: set[str]) -> list[str]:
    """校验题库，返回问题列表（ERROR/WARN 前缀区分严重程度）。"""
    issues: list[str] = []
    seen_ids: dict[str, str] = {}
    seen_questions: dict[str, str] = {}

    for q in questions:
        qid = q.get('id', '(缺id)')
        for field, expected_type in REQUIRED_FIELDS.items():
            if field not in q:
                issues.append(f'ERROR {qid}: 缺字段 {field}')
            elif not isinstance(q[field], expected_type):
                issues.append(f'ERROR {qid}: 字段 {field} 类型应为 {expected_type.__name__}')
        if qid in seen_ids:
            issues.append(f'ERROR {qid}: id 重复（另有 {seen_ids[qid]}）')
        seen_ids[qid] = qid

        norm = normalize_text(q.get('question', ''))
        if norm in seen_questions:
            issues.append(f'ERROR {qid}: 与 {seen_questions[norm]} 问题重复')
        seen_questions[norm] = qid

        golds = q.get('gold_sources') or []
        for gold in golds:
            if gold not in valid_titles:
                issues.append(f'ERROR {qid}: gold_sources 中的「{gold}」不在知识库 file_title 清单里')

        if q.get('expect_fallback') and golds:
            issues.append(f'WARN  {qid}: expect_fallback=true 却填了 gold_sources {golds}')

        overlap = set(q.get('must_have') or []) & set(q.get('must_not') or [])
        if overlap:
            issues.append(f'ERROR {qid}: must_have 与 must_not 冲突 {sorted(overlap)}')

        if not q.get('evidence'):
            issues.append(f'WARN  {qid}: evidence 为空，人工无法核对答案依据')
        if len(q.get('ground_truth') or '') < 10:
            issues.append(f'WARN  {qid}: ground_truth 过短（{len(q.get("ground_truth") or "")} 字）')
    return issues


def write_jsonl(questions: list[dict]) -> None:
    questions = sorted(questions, key=lambda q: (CATEGORY_ORDER.index(q['category'])
                                                 if q.get('category') in CATEGORY_ORDER else 99, q['id']))
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open('w', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')


def write_review_md(questions: list[dict]) -> Path:
    """生成人工评审表：逐题列出问题、参考答案、来源、断言与原文依据。"""
    out = DATASET_DIR / 'review_questions_v1.md'
    lines: list[str] = ['# 评估题库评审表 v1', '']
    lines.append(f'共 {len(questions)} 题。请逐题检查：**问题问得是否合理**、**参考答案是否与原文一致**、'
                 '**断言词（must_have / must_not）是否恰当**。修改后告诉我，我据此重新冻结 jsonl。')
    lines.append('')
    lines.append('| 类别 | 题数 |')
    lines.append('| --- | --- |')
    for cat in CATEGORY_ORDER:
        n = sum(1 for q in questions if q.get('category') == cat)
        lines.append(f'| {cat} | {n} |')
    lines.append('')

    current_cat = None
    for q in sorted(questions, key=lambda q: (CATEGORY_ORDER.index(q['category'])
                                             if q.get('category') in CATEGORY_ORDER else 99, q['id'])):
        if q.get('category') != current_cat:
            current_cat = q.get('category')
            lines.append(f'## {current_cat}')
            lines.append('')
        flags = []
        if q.get('expect_fallback'):
            flags.append('无资料题（应走兜底）')
        if q.get('expect_compliance'):
            flags.append('合规诱导题')
        lines.append(f"### {q['id']}　{'　'.join(f'`{f}`' for f in flags)}")
        lines.append('')
        if q.get('known_issue'):
            lines.append(f"- **⚠ 已知失败题（保留在题库中作对照）**：{q['known_issue']}")
        lines.append(f"- **问题**：{q['question']}")
        lines.append(f"- **参考答案**：{q['ground_truth']}")
        lines.append(f"- **标准来源**：{'、'.join(q['gold_sources']) or '（无，预期兜底）'}")
        lines.append(f"- **应出现关键词**：{'、'.join(q['must_have']) or '—'}")
        lines.append(f"- **违规词**：{'、'.join(q['must_not']) or '—'}")
        lines.append(f"- **出题意图**：{q.get('design_note', '')}")
        lines.append(f"- **语料依据**：{q.get('evidence', '')}")
        lines.append('')
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='题库构建与校验')
    parser.add_argument('--from-jsonl', action='store_true',
                        help='不合并 drafts，直接校验已冻结的 questions_v1.jsonl 并重建评审表')
    args = parser.parse_args()

    if args.from_jsonl:
        questions = [json.loads(line) for line in DATASET_PATH.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        questions = load_drafts()

    titles = kb_file_titles()
    issues = validate(questions, titles)
    errors = [i for i in issues if i.startswith('ERROR')]
    warns = [i for i in issues if i.startswith('WARN')]

    print(f'题库总数：{len(questions)}｜KB 资料数：{len(titles)}')
    for cat in CATEGORY_ORDER:
        print(f'  {cat}: {sum(1 for q in questions if q.get("category") == cat)} 题')
    print(f'\n校验：{len(errors)} 个 ERROR，{len(warns)} 个 WARN')
    for i in errors + warns:
        print('  ' + i)

    if errors:
        print('\n存在 ERROR，未写出题库文件，请先修正 drafts。')
        sys.exit(1)

    if not args.from_jsonl:
        write_jsonl(questions)
    review = write_review_md(questions)
    print(f'\n已写出：{DATASET_PATH}\n评审表：{review}')
