"""
    @Desc   :
    @Time   :2026/9/3 15:12
    @Author :爱吃肯德基
"""
from typing import Any

from pymilvus import DataType

from src.common.config.milvus_config import milvus_config
from src.common.logging.logger import node_log, logger, step_log
from src.utils.task_utils import add_running_task, add_done_task
from src.processor.import_processor.state import ImportGraphState
from src.utils.clients.milvus_utils import get_milvus_client


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: ImportGraphState) -> list[dict[str, Any]]:
    """
      获取数据,并且校验
    :param state:
    :return:
    """
    # 1. 获取数据
    embeddings_content: list[dict[str, Any]] = state.get("embeddings_content")
    # 2. 数据校验
    if not embeddings_content:
        logger.error(f"embeddings_content没有值,业务无法继续,提前终止!")
        raise ValueError(f"embeddings_content没有值,业务无法继续,提前终止!")
    # 3. 返回结果
    return embeddings_content


@step_log("step_2_prepared_item_name_collection")
def step_2_prepared_item_name_collection():
    """
    创建item_name对应的集合的
    :return:
    """
    milvus_client = get_milvus_client()

    # 检查是否存在集合
    has_collection = milvus_client.has_collection(collection_name=milvus_config.chunks_collection)
    if has_collection:
        logger.info(f"{milvus_config.chunks_collection}已经存在,可以直接使用!")
        return
    logger.info(f"{milvus_config.chunks_collection}不存在,进行集合的创建!")
    # 1.创建schema
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
    )
    # 数据类型: https://milvus.io/docs/zh/number.md  datatype = DataType -> milvus . 类型
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR,
                     dim=1024)  # 稠密向量和嵌入式模型有关系 1. 生成的浮点类型 16 32 8 2. 维度
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    # 2.创建索引
    index_params = milvus_client.prepare_index_params()
    # 稠密向量
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",  # FLAT 全盘搜索 IVF_FLAT 分桶,每个有中心点 效率快,准确度稍低 HNSW 分层图导航 -> 地图 -> 精准 | 效率低
        index_name="dense_vector_index",
        params={
            "M": 64,  # 相邻最大的节点数量
            "efConstruction": 100  # 候选的节点数量
        },
        metric_type="COSINE"  # 稠密推荐 COSINE / IP / L2
    )
    # 稀疏
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",  # 倒排索引  使用 index -> 向量 检索 检索特性index关联向量!
        index_name="sparse_vector_index",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},  # 跳过低分! 计算高分数据集!!
        metric_type="IP"  # 稀疏推荐 IP[关心关键词命中,也会考虑向量的相似度] / BM25 [只关注关键字的命中]-> milvus 3.0
    )

    # 3.创建集合
    milvus_client.create_collection(
        collection_name=milvus_config.chunks_collection,
        schema=schema,
        index_params=index_params
    )

    milvus_client.load_collection(collection_name=milvus_config.chunks_collection)


@step_log("step_3_insert_item_name_data")
def step_3_insert_item_name_data(embeddings_content: list[dict]):
    """
    插入数据
    :param item_name:
    :param file_title:
    :return:
    """
    file_title: str = embeddings_content[0].get("file_title")
    # 2. 先删除数据
    # 文档 -> 集合 -> 先删除旧的数据 -> file_title | item_name
    milvus_client = get_milvus_client()
    milvus_client.delete(
        collection_name=milvus_config.chunks_collection,
        filter=f"file_title == '{file_title}'"  # mysql where  坑1: milvus的等于 ==  坑2: 会值变成列名
    )
    # 3. 插入数据
    milvus_client.insert(
        collection_name=milvus_config.chunks_collection,
        data=embeddings_content
    )


@node_log('node_import_milvus')
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点：导入向量数据库
    """
    add_running_task(state.get('task_id'), 'node_import_milvus')

    # 参数获取并校验
    embeddings_content = step_1_validate_and_get_data(state)

    # 创建对应的collection
    step_2_prepared_item_name_collection()

    # 将数据导入collection
    step_3_insert_item_name_data(embeddings_content)

    logger.info(
        f"{embeddings_content[0].get("file_title")}文档对应的数据已经完成向量数据库的导入,导入的数量:{len(embeddings_content)}!")

    add_done_task(state.get('task_id'), 'node_import_milvus')
    return state


if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "item_name": "测试项目_Milvus",
        "file_title": "test.pdf",
        "embeddings_content": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
            ,
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.2] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 执行节点函数
        result_state = node_import_milvus(test_state)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
