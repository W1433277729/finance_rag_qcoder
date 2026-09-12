"""
    @Desc   :
    @Time   :2026/9/3 15:11
    @Author :爱吃肯德基
"""
import json
from pathlib import Path
from typing import Any

from src.common.logging.logger import node_log, logger, step_log
from src.utils.task_utils import add_running_task, add_done_task
from src.processor.import_processor.state import ImportGraphState
from src.utils.lm.embedding_utils import generate_embeddings

# 向量化批次大小：每批处理 5 条切片，避免显存溢出
EMBEDDING_BATCH_SIZE = 5


@step_log('step_1_validate_and_get_data')
def step_1_validate_and_get_data(state: ImportGraphState):
    # 1. 获取数据
    chunks: list[dict[str, Any]] = state.get("chunks")
    # 2. 数据校验
    if not chunks:
        # 核心参数错了
        md_path: str = state.get("md_path")
        md_path_obj: Path = Path(md_path)
        if md_path_obj.is_file():
            json_path_obj: Path = md_path_obj.with_name(f"{md_path_obj.stem}.json")
            if json_path_obj.is_file():
                chunks = json.loads(json_path_obj.read_text(encoding="utf-8"))
                for chunk in chunks:
                    chunk['item_name'] = state.get("item_name")
                state['chunks'] = chunks
            else:
                logger.error(f"chunks没有值,json备份文件不存在,抛出异常!")
                raise ValueError(f"chunks没有值,json备份文件不存在,抛出异常!")
        else:
            logger.error(f"chunks没有值,同时也没有读取到对应json备份数据,抛出异常!")
            raise ValueError(f"chunks没有值,同时也没有读取到对应json备份数据,抛出异常!")
    # 3. 返回结果
    return chunks


@step_log('step_2_generate_vector')
def step_2_batch_generate_vector(chunks: list[dict[str, Any]]):
    """
    经过嵌入模型，将chunks 转为嵌入向量
    :param chunks:
    """
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        current_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
        contents = [f'{chunk.get('item_name')}\n{chunk.get('content')}' for chunk in current_chunks]
        embeddings = generate_embeddings(contents)

        dense_vector_list = embeddings['dense']
        sparse_vector_list = embeddings['sparse']

        # 将稀稠向量保存至 对应的chunk中
        for index, chunk in enumerate(current_chunks):
            chunk['dense_vector'] = dense_vector_list[index]
            chunk['sparse_vector'] = sparse_vector_list[index]


@node_log('node_bge_embedding')
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点：切块向量化节点
    """
    add_running_task(state.get('task_id'), 'node_bge_embedding')

    # 参数获取并校验
    chunks = step_1_validate_and_get_data(state)

    # 对chunks 批量生成向量
    step_2_batch_generate_vector(chunks)

    # 更新state
    state['embeddings_content'] = chunks

    add_done_task(state.get('task_id'), 'node_bge_embedding')
    return state


if __name__ == '__main__':
    # 构造模拟测试状态：模拟上游节点输出的chunks数据，贴合真实业务场景
    test_state = ImportGraphState({
        "task_id": "test_task_embedding_001",  # 测试任务ID
        "chunks": [  # 模拟带item_name的文本切片（上游商品名称识别节点产出）
            {
                "content": "这是一个测试文档的内容，用于验证向量化是否成功。",
                "title": "测试文档标题",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            },
            {
                "content": "这是第二个测试文档的内容，用于验证批量处理逻辑。",
                "title": "测试文档标题2",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            }
        ]
    })

    # 执行本地测试
    logger.info("=== BGE-M3向量化节点本地单元测试启动 ===")
    try:
        # 调用核心节点函数
        result_state = node_bge_embedding(test_state)
        # 提取测试结果
        result_chunks = result_state.get("chunks", [])

        # 打印测试结果统计
        logger.info(f"=== 向量化节点本地测试完成 ===")
        logger.info(f"测试任务ID：{test_state.get('task_id')}")
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")
        logger.info(f"返回的结果:{result_chunks}")

    except Exception as e:
        logger.error(f"=== 向量化节点本地测试失败 ===" f"错误原因：{str(e)}", exc_info=True)
        # 新手友好提示：给出核心排查方向
        logger.warning("排查提示：请检查BGE-M3模型路径、显存是否充足、环境变量配置是否正确")
