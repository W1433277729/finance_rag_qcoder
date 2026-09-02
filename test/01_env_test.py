"""
    @Desc   :
    @Time   :2026/9/2 16:46
    @Author :爱吃肯德基
"""
# test/01_env_test.py
import os
from dotenv import load_dotenv

load_dotenv(
    override=True
)

print(os.getenv("BGE_M3_PATH"))
# load_dotenv(override=True) → 输出 dotenv_val（.env覆盖系统）