import threading

from FlagEmbedding import FlagReranker
from src.common.config.reranker_config import reranker_config

_reranker_model = None
# 与 embedding 单例同理：并发首次调用会重复加载同一模型，加锁串行化（当前调用方是单线程，
# 但查询服务并发接待多个会话时同样有这个风险）
_reranker_lock = threading.Lock()

def get_reranker_model():
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model
    with _reranker_lock:
        # 双重检查：等锁期间可能已被其它线程加载完
        if _reranker_model is None:
            _reranker_model= FlagReranker(
                model_name_or_path=reranker_config.bge_reranker_large,
                device=reranker_config.bge_reranker_device,
                use_fp16=reranker_config.bge_reranker_fp16
            )
    return _reranker_model

def compute_token_number( content: str):
    """
      计算字符串对应的token的数量
    :param content:
    :return:
    """
    reranker_model = get_reranker_model()
    tokenizer = reranker_model.tokenizer
    token_list = tokenizer.encode(content, add_special_tokens=False)
    return len(token_list)