"""
    @Desc   : RAGAS embeddings 适配器：把项目本地 BGE-M3 包装成 ragas 的 BaseRagasEmbedding，
              用于 AnswerRelevancy / AnswerCorrectness 等需要向量相似度的指标
    @Time   : 2026/9/22
    @Author :爱吃肯德基
"""
import asyncio

from eval import ragas_compat  # noqa: F401  必须先于 ragas 导入（兼容补丁）
from ragas.embeddings import BaseRagasEmbedding

from src.common.logging.logger import logger
from src.utils.lm.embedding_utils import generate_embeddings

# 项目 BGE-M3 的稠密向量维度（与 .env 的 EMBEDDING_DIM 一致）
EXPECTED_DIM = 1024


class BgeM3RagasEmbedding(BaseRagasEmbedding):
    """
    复用项目既有的本地 BGE-M3（embedding_utils.generate_embeddings），
    只取稠密向量（ragas 的相似度指标不需要稀疏向量）。
    """

    def embed_text(self, text: str, **kwargs) -> list[float]:
        if not text:
            # 空文本会让 BGE-M3 报错，返回零向量占位（相似度指标对此不敏感）
            return [0.0] * EXPECTED_DIM
        return generate_embeddings([text])['dense'][0]

    def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        if not texts:
            return []
        return generate_embeddings(list(texts))['dense']

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        # BGE-M3 是同步的本地模型，放到线程里避免阻塞判官的事件循环
        return await asyncio.to_thread(self.embed_text, text, **kwargs)

    async def aembed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_texts, texts, **kwargs)


def get_ragas_embeddings() -> BgeM3RagasEmbedding:
    """构建 ragas 可用的 embeddings 实例（本地 BGE-M3，不产生外部 API 调用）。"""
    logger.info('已接入本地 BGE-M3 作为 ragas embeddings')
    return BgeM3RagasEmbedding()
