"""
    @Desc   :
    @Time   :2026/9/3 11:45
    @Author :爱吃肯德基
"""
import sys
from pathlib import Path

# --------------------------
# 路径兜底：确保「项目根目录」在 sys.path 中
# 本项目内部模块统一使用 src.xxx 绝对导入，本文件在 src 包首次被导入时执行，
# 顺势把项目根目录补进 sys.path，避免因运行目录不同导致 import src.xxx 失败。
# --------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))