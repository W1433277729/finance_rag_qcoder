"""
    @Desc   :
    @Time   :2026/9/3 15:11
    @Author :爱吃肯德基
"""
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType, MilvusClient

from src.common.config.milvus_config import milvus_config
from src.common.logging.logger import node_log, logger, step_log
from src.utils.task_utils import add_running_task, add_done_task
from src.processor.import_processor.state import ImportGraphState
from src.utils.clients import milvus_utils
from src.utils.lm import lm_utils
from src.utils.lm.embedding_utils import generate_embeddings
from src.utils.load_prompt import load_prompt

# 主体识别上下文切片数：取前 K 个切片用于 LLM 识别
ITEM_NAME_CONTEXT_CHUNK_K = 5
# 主体识别上下文总字符数上限：防止上下文过长导致大模型输入超限
ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS = 2000

COLLECTION_NAME = milvus_config.item_name_collection


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: ImportGraphState) -> tuple[list[dict[str, Any]], str]:
    """
      获取数据,并且校验
    :param state:
    :return:
    """
    # 1. 获取数据
    chunks: list[dict[str, Any]] = state.get("chunks")
    file_title: str = state.get("file_title")
    # 2. 数据校验
    if not chunks:
        # 核心参数错了
        md_path: str = state.get("md_path")
        md_path_obj: Path = Path(md_path)
        if md_path_obj.is_file():
            json_path_obj: Path = md_path_obj.with_name(f"{md_path_obj.stem}.json")
            if json_path_obj.is_file():
                chunks = json.loads(json_path_obj.read_text(encoding="utf-8"))
                state['chunks'] = chunks
            else:
                logger.error(f"chunks没有值,json备份文件不存在,抛出异常!")
                raise ValueError(f"chunks没有值,json备份文件不存在,抛出异常!")
        else:
            logger.error(f"chunks没有值,同时也没有读取到对应json备份数据,抛出异常!")
            raise ValueError(f"chunks没有值,同时也没有读取到对应json备份数据,抛出异常!")
    if not file_title:
        md_path: str = state.get("md_path")
        md_path_obj: Path = Path(md_path)
        file_title = md_path_obj.stem
        state['file_title'] = file_title
        logger.warning(f"file_title不存在,给与默认值:{file_title}")
    # 3. 返回结果
    return chunks, file_title


@step_log("step_2_call_llm_return_item_name")
def step_2_call_llm_return_item_name(chunks: list[dict[str, Any]], file_title) -> str:
    """
    调用模型生成 文档主体内容
    :param chunks:
    :param file_title:
    """
    # 创建模型
    model = lm_utils.get_llm_client()
    # 构建提示词
    context = ''
    for chunk in chunks[:ITEM_NAME_CONTEXT_CHUNK_K]:
        context += f'标题:{chunk.get('parent_title')},内容:{chunk.get('content')}\n'
    context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]
    prompt_text = load_prompt('item_name_recognition', file_title=file_title, context=context)
    message = HumanMessage(content=prompt_text)
    # 拼接链
    chains = model | StrOutputParser()
    item_name = chains.invoke([message])
    if not item_name:
        item_name = file_title
        logger.warning(f"没有识别出item_name,使用file_title赋值:{item_name}")
    return item_name


@step_log("step_3_padding_item_name_to_chunks")
def step_3_padding_item_name_to_chunks(chunks: list[dict[str, Any]], item_name):
    """
    把 item_name 放到chunks中
    :param chunks:
    :param item_name:
    """
    for chunk in chunks:
        chunk['item_name'] = item_name


@step_log("step_4_prepared_item_name_collection")
def step_4_prepared_item_name_collection():
    milvus_client = milvus_utils.get_milvus_client()
    # 检查是否存在集合
    has_collection = milvus_client.has_collection(collection_name=milvus_config.item_name_collection)
    if has_collection:
        logger.info(f"{milvus_config.item_name_collection}已经存在,可以直接使用!")
        return
    logger.info(f"{milvus_config.item_name_collection}不存在,进行集合的创建!")
    # 1.创建schema

    # 创建 schema
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=False, )
    schema.add_field(field_name='pk', datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name='file_title', datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name='item_name', datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)
    # 设置索引参数
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name='dense_vector',
        index_type='HNSW',
        index_name='dense_vector_index',
        metric_type='COSINE',
        params={
            "M": 64,  # 相邻最大的节点数量
            "efConstruction": 100  # 候选的节点数量
        }
    )
    index_params.add_index(
        field_name='sparse_vector',
        index_type='SPARSE_INVERTED_INDEX',
        index_name='sparse_vector_index',
        metric_type='IP',  # 稀疏推荐 IP[关心关键词命中,也会考虑向量的相似度] / BM25 [只关注关键字的命中]-> milvus 3.0
        params={"inverted_index_algo": "DAAT_MAXSCORE"}  # 跳过低分! 计算高分数据集!!
    )

    # 创建 collection
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
        schema=schema
    )


@step_log("step_5_insert_item_name_data")
def step_5_insert_item_name_data(item_name, file_title):
    # 调用嵌入模型，生成嵌入向量:包含稠密和稀疏向量
    embedding = generate_embeddings([item_name])
    # result = {"dense":[[]],"sparse":[{}]}
    dense_vector = embedding['dense'][0]
    sparse_vector = embedding['sparse'][0]

    milvus_client: MilvusClient = milvus_utils.get_milvus_client()
    # 先删除数据
    # 文档 -> 集合 -> 先删除旧的数据 -> file_title | item_name
    milvus_client.delete(
        collection_name=milvus_config.item_name_collection,
        filter=f"file_title == '{file_title}'"  # mysql where  坑1: milvus的等于 ==  坑2: 会值变成列名
    )
    data = {
        'item_name': item_name,
        'file_title': file_title,
        'dense_vector': dense_vector,
        'sparse_vector': sparse_vector
    }

    res = milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
    logger.info(f'插入成功, {res}')


@node_log('node_item_recognition')
def node_item_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点：文档主体识别节点
    """
    add_running_task(task_id=state.get('task_id'), node_name='node_item_recognition')
    # ======================== step1 参数获取及校验 =======================
    chunks, file_title = step_1_validate_and_get_data(state)

    # ======================== step2 使用模型提取 主体信息 ======================
    item_name: str = step_2_call_llm_return_item_name(chunks, file_title)

    # ======================== step3 回填数据，修改chunks内容 ===================
    step_3_padding_item_name_to_chunks(chunks, item_name)

    # ======================== step4 创建集合 ============================
    step_4_prepared_item_name_collection()

    # ======================== step5 存储数据 ============================
    step_5_insert_item_name_data(item_name, file_title)

    # ======================== step6 更新状态 ============================
    state['chunks'] = chunks
    state['item_name'] = item_name
    add_done_task(task_id=state.get('task_id'), node_name='node_item_recognition')

    return state


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    """
        商品名称识别节点本地测试方法
        功能：模拟LangGraph流程输入，独立测试node_item_recognition节点全链路逻辑
        适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
        测试前准备：
            1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
            2. 确保大模型、Milvus、BGE-M3服务均可正常访问
            3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
        使用方法：
            直接运行该函数：if __name__ == "__main__": test_node_item_recognition()
        """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })
        # 2. 调用商品名称识别核心节点
        result_state = node_item_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)
