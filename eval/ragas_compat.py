"""
    @Desc   : ragas 导入兼容补丁（上游问题，非本项目代码瑕疵）
    @Time   : 2026/9/22
    @Author :爱吃肯德基

背景：ragas 0.4.3 在 ragas/llms/base.py 里 `import langchain_community.chat_models.vertexai`，
而本项目锁定的 langchain-community 0.4.x 已移除该模块，导致 `import ragas` 直接 ImportError。
这里注入一个最小桩模块让 ragas 完成导入；本项目不使用 VertexAI，无功能影响。

用法：任何要 import ragas 的文件，都必须先 `from eval import ragas_compat  # noqa: F401`。
（本模块在被 import 时就执行补丁，无需显式调用函数。）
"""
import sys
import types

from src.common.logging.logger import logger

_MODULE_NAME = 'langchain_community.chat_models.vertexai'


def _patch() -> None:
    if _MODULE_NAME in sys.modules:
        return
    try:
        from langchain_community.chat_models import vertexai  # noqa: F401
        return
    except Exception:
        pass

    stub = types.ModuleType(_MODULE_NAME)

    class ChatVertexAI:  # 仅占位，不会被实例化
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[_MODULE_NAME] = stub
    logger.warning(f'ragas 兼容补丁已生效：注入桩模块 {_MODULE_NAME}')


_patch()
