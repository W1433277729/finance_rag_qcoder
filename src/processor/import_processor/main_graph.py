"""
    @Desc   :导入流程主图代码
    @Time   :2026/9/2 19:32
    @Author :爱吃肯德基
"""
import json

from dotenv import load_dotenv
from src.common.logging.logger import logger
from langgraph.graph import END, StateGraph
from src.processor.import_processor.nodes import node_entry, node_pdf_to_md, node_md_img, node_document_split, \
    node_bge_embedding, node_import_milvus, node_item_name_recognition
from src.processor.import_processor.state import create_default_state, ImportGraphState

# 初始化环境变量：必须配置在读取前操作，确保后续节点能获取到环境变量中的信息
load_dotenv()

# ===================== 1. 初始化LangGraph状态图 =====================
# 核心：StateGraph是LangGraph的核心类，用于构建有状态的工作流
# 参数ImportGraphState：自定义TypedDict类型，定义了工作流的**全量状态字段**
# 作用：所有节点的入参都是该状态对象，节点返回的键值对会自动合并回状态，实现节点间数据共享
workflow = StateGraph(ImportGraphState)

# ===================== 2. 注册所有业务节点 =====================
# 语法：add_node("节点唯一标识", 节点函数)
# 要求：节点函数必须接收「状态对象」作为入参，返回字典（用于更新状态）
# 所有节点按「知识库导入流程」先后顺序注册，节点标识与函数名保持一致，便于维护
workflow.add_node('node_entry', node_entry)  # 流程入口：参数初始化、输入校验
workflow.add_node('node_pdf_to_md', node_pdf_to_md)  # pdf 转 md: pdf文档的前置处理
workflow.add_node('node_md_img', node_md_img)  # md文档中的图片处理：保证图片的可访问性
workflow.add_node('node_document_split', node_document_split)  # 文档分块：解决大文本无法向量发/推理的问题
workflow.add_node('node_item_name_recognition', node_item_name_recognition)  # 文本块主体识别：业务定制化步骤，提取核心业务标识
workflow.add_node('node_bge_embedding', node_bge_embedding)  # 文本块向量化：为 Milvus 存储
workflow.add_node('node_import_milvus', node_import_milvus)  # 向量入库：将向量数据持久化到Milvus中

# ===================== 3. 设置工作流入口节点 =====================
# 语法：set_entry_point("节点标识") → 推荐写法，直接指定流程起始节点
# 等效写法：workflow.add_edge(START, "node_entry")（START是LangGraph内置起始常量）
# 作用：指定工作流执行的第一个节点，替代手动添加START到目标节点的边，代码更简洁
workflow.set_entry_point('node_entry')


# ===================== 4. 定义条件路由函数（入口节点后的分支逻辑） =====================
# 核心：根据状态中的配置项，动态决定后续执行路径，实现「PDF导入」/「MD直接导入」分支
# 要求：接收状态对象为入参，返回「目标节点标识」或END（内置结束常量）
def route_after_entry(state: ImportGraphState) -> str:
    """
    入口节点后的条件路由逻辑
    :param state: 工作流全量状态对象，包含所有配置项和中间结果
    :return: 目标节点标识/END，LangGraph会自动跳转至对应的节点
    """
    if state.get('is_md_read_enabled'):
        # 分支1：开启MD直接导入 → 跳过PDF转MD，直接执行MD图片处理
        return 'node_md_img'
    elif state.get('is_pdf_read_enabled'):
        # 分支2：开启PDF导入 → 执行PDF转MD，再走后续流程
        return 'node_pdf_to_md'
    else:
        # 分支3：未开启任何导入配置 → 直接终止工作流（END是LangGraph内置结束常量）
        return END


# 注册条件边：将入口节点与路由函数绑定
# 语法：add_conditional_edges("源节点标识", 路由函数,path_map[可选])
# 什么时候可以省略：路由函数返回的字符串刚好等于目标节点名称，可以省略 path_map
# 什么时候不能省略：1. 路由函数返回的字符串不等于节点名称的时候 2. 如果你想要显示的视图路径必须显示路径
# 作用：源节点执行完成后，调用路由函数，根据返回值动态跳转到目标节点
workflow.add_conditional_edges(
    'node_entry',
    route_after_entry,
    {
        'node_md_img': 'node_md_img',
        'node_pdf_to_md': 'node_pdf_to_md',
        END: END
    }
)

# ===================== 5. 注册静态顺序边（分支合并后的统一流程） =====================
# 核心：所有分支最终合并为「固定顺序执行流程」，从MD图片处理到知识图谱入库，一步到底
# 语法：add_edge("源节点标识", "目标节点标识/END") → 静态边，固定路由关系，无分支逻辑
workflow.add_edge('node_pdf_to_md', 'node_md_img')  # pdf转md -> md图片处理
workflow.add_edge('node_md_img', 'node_document_split') # md图片处理 -> 文档切分
workflow.add_edge('node_document_split', 'node_item_name_recognition')  # 文档切分 -> 文本块主体识别
workflow.add_edge('node_item_name_recognition', 'node_bge_embedding')   # 文本块主体识别 -> 文本块向量化
workflow.add_edge('node_bge_embedding', 'node_import_milvus')   # 文本块数据向量化 -> 向量数据持久化
workflow.add_edge('node_import_milvus', END)    # 向量数据持久化 -> END

# ===================== 6. 编译工作流为可执行对象 =====================
# 语法：compile() → 将StateGraph构建的流程编译为LangGraph的可执行应用
# 作用：生成可调用的kb_import_app，通过invoke()方法触发工作流执行
# 特性：编译后可重复调用，支持传入不同的初始状态，实现多任务执行

kb_import_app = workflow.compile()

if __name__ == '__main__':
    logger.info('-----main_graph 开始测试-------')
    init_state = create_default_state(local_file_path = '万用表RS-12的使用.pdf')
    final_state = None

    kb_import_app.invoke(init_state)
    # 格式化输出最终状态
    logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False)}")
    logger.info('图结构')
    kb_import_app.get_graph().print_ascii()