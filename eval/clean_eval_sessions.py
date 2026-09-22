"""
    @Desc   : 清理评估产生的 Mongo 对话历史（session_id 以 eval_ 开头），默认只预览不删除
    @Time   : 2026/9/22
    @Author :爱吃肯德基
"""
import argparse
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.logging.logger import logger
from src.utils.clients.mongo_history_utils import get_history_mongo_tool

# 默认只清理评估会话前缀，避免误删真实用户会话
DEFAULT_PREFIX = 'eval_'


def collect_sessions(pattern: str) -> list[tuple[str, int]]:
    """按正则前缀统计会话及其消息数。"""
    tool = get_history_mongo_tool()
    pipeline = [
        {'$match': {'session_id': {'$regex': pattern}}},
        {'$group': {'_id': '$session_id', 'count': {'$sum': 1}}},
        {'$sort': {'_id': 1}},
    ]
    return [(row['_id'], row['count']) for row in tool.chat_message.aggregate(pipeline)]


def build_pattern(prefix: str, run_id: str | None) -> str:
    """
    构造会话匹配正则：始终限定在 prefix（默认 eval_）之内，
    再按 run_id 做「包含」匹配——实际 session_id 形如 eval_<时间戳>_<标签>_<题号>，
    因此传 --run-id full_v1 即可命中该次运行的全部题。
    """
    pattern = f'^{re.escape(prefix)}'
    if run_id:
        pattern += f'.*{re.escape(run_id)}'
    return pattern


def main() -> None:
    parser = argparse.ArgumentParser(description='清理评估产生的 Mongo 会话历史')
    parser.add_argument('--prefix', default=DEFAULT_PREFIX, help=f'会话前缀，默认 {DEFAULT_PREFIX}')
    parser.add_argument('--run-id', default=None,
                        help='只清理 session_id 包含该字符串的会话（如 --run-id full_v1），默认清理该前缀下全部')
    parser.add_argument('--yes', action='store_true', help='确认执行删除；不加此参数只预览')
    args = parser.parse_args()

    pattern = build_pattern(args.prefix, args.run_id)
    sessions = collect_sessions(pattern)
    total = sum(count for _, count in sessions)
    print(f'匹配规则：/{pattern}/｜会话数：{len(sessions)}｜消息数：{total}')
    for session_id, count in sessions[:20]:
        print(f'  {count:>4} 条  {session_id}')
    if len(sessions) > 20:
        print(f'  ...（其余 {len(sessions) - 20} 个会话省略）')

    if not sessions:
        return
    if not args.yes:
        print('\n当前为预览模式，未删除任何数据。确认后加 --yes 执行删除。')
        return

    tool = get_history_mongo_tool()
    result = tool.chat_message.delete_many({'session_id': {'$regex': pattern}})
    logger.info(f'已删除评估会话历史，规则=/{pattern}/，删除消息数={result.deleted_count}')


if __name__ == '__main__':
    main()
