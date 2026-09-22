"""
    @Desc   : 盘点 knowledge base 实际入库内容（只读）：按 file_title 统计切片数与元数据分布，
              供评估集编写时确认 gold_sources 的合法取值域
    @Time   : 2026/9/22
    @Author :爱吃肯德基
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.config.milvus_config import milvus_config
from src.utils.clients.milvus_utils import get_milvus_client

# 单次 query 上限，超过则视为截断并告警
QUERY_LIMIT = 16384


def load_all_chunks() -> list[dict]:
    """读取 kb_chunks 全部切片的轻量字段（不含向量）。"""
    client = get_milvus_client()
    if client is None:
        raise RuntimeError('Milvus 客户端初始化失败，请确认 MILVUS_URL 与 Milvus 服务状态')
    collection = milvus_config.chunks_collection
    client.load_collection(collection_name=collection)
    rows = client.query(
        collection_name=collection,
        filter='chunk_id >= 0',
        output_fields=['file_title', 'content_type', 'product_name', 'product_code',
                       'institution_name', 'risk_level', 'publish_date', 'source_file'],
        limit=QUERY_LIMIT,
    )
    return rows or []


def summarize(rows: list[dict]) -> None:
    by_file = Counter()
    by_file_meta = defaultdict(set)
    by_content_type = Counter()
    for r in rows:
        title = r.get('file_title') or '(空)'
        by_file[title] += 1
        by_content_type[r.get('content_type') or '(空)'] += 1
        for field in ('content_type', 'product_name', 'product_code', 'risk_level', 'publish_date'):
            if r.get(field):
                by_file_meta[title].add((field, r[field]))

    print(f'集合：{milvus_config.chunks_collection}｜数据库：{milvus_config.database}')
    print(f'切片总数：{len(rows)}｜资料数：{len(by_file)}'
          f'{"（已达到单次查询上限，结果可能被截断）" if len(rows) >= QUERY_LIMIT else ""}')
    print('\n=== 按资料（file_title）===')
    for title, n in sorted(by_file.items(), key=lambda kv: -kv[1]):
        metas = '；'.join(f'{k}={v}' for k, v in sorted(by_file_meta.get(title, set())))
        print(f'{n:>4} 片  {title}')
        if metas:
            print(f'        元数据：{metas}')
    print('\n=== 按 content_type ===')
    for ct, n in sorted(by_content_type.items(), key=lambda kv: -kv[1]):
        print(f'{n:>4} 片  {ct}')


if __name__ == '__main__':
    summarize(load_all_chunks())
