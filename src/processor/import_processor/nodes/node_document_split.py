"""
    @Desc   :
    @Time   :2026/9/3 15:10
    @Author :爱吃肯德基
"""
import json
import os
import re
from pathlib import Path
from typing import Tuple, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.logging.logger import step_log
from src.common.logging.logger import node_log, logger
from src.utils.task_utils import add_running_task, add_done_task
from src.processor.import_processor.state import ImportGraphState

# ====================== 全局配置（可根据模型调整）======================
# 文本切块最大长度：单个文本块最多包含 1000 字符（防止过长导致向量失真）
CHUNK_MAX_SIZE = 1000
# 文本切块基准长度：单个文本块理想大小为 600 字符（兼顾语义完整性 + 检索精度）
CHUNK_SIZE = 600
# 文本块重叠长度：相邻块之间重叠 20 字符，保证语义不被切断、上下文连贯
CHUNK_OVERLAP = 50
# 最小碎片阈值：低于这个长度判定为短碎片，需要尝试合并
CHUNK_MIN = 400


@step_log('step_1_validate_get_data')
def step_1_validate_get_data(state: ImportGraphState) -> Tuple[str, str, str]:
    """
    校验并获取参数，进行简单清洗
    :param state:
    """
    md_content: str = state.get("md_content")
    file_title: str = state.get("file_title")
    md_path: str = state.get("md_path")
    # 非空校验、空值更新
    if not md_content:
        if (not md_path) or (not Path(md_path).is_file()):
            # md_path为空,或者是不存在(不是文件)
            logger.error(f"md_content内容为空,md_path也为空或者没有对应的文件,业务无法继续,提前终止!")
            raise ValueError(f"md_content内容为空,md_path也为空或者没有对应的文件,业务无法继续,提前终止!")
        md_content = Path(md_path).read_text(encoding="utf-8")
        state["md_content"] = md_content
    if not file_title:
        file_title = Path(md_path).stem
        logger.warning(f"file_title为空,给与默认值:{file_title}")
        state["file_title"] = file_title
    # 清洗和替换数据 统一不同的系统 换成 \n
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    # 3.返回对应的结果
    return md_content, file_title, md_path


@step_log("step_2_split_document_by_title")
def step_2_split_document_by_title(md_content: str, file_title: str) -> list[dict[str, Any]]:
    # 存储最终返回的切块列表
    chunks: list[dict[str, Any]] = []

    # 当前激活的完整嵌套标题  （例如：#标题1_##标题2）
    current_title: str | None = None

    # chunk正文缓存区  存放当前标题下的正文与代码行，不包含标题行本身
    current_content_lines: list[str] = []

    # 孤儿数据缓存区   第一个标题出现前的所有文本行
    orphan_lines: list[str] = []

    # 标记位： 记录开头累积的孤儿数据是否已经被成功合并到了首个chunk中
    has_flushed_first_chunk: bool = False

    # 状态标记  标记当前行是否处于代码块内部(```或者~~~)
    is_code: bool = False

    # 多级标题栈: 用于标记H1~H6   例如： ["H1","H2",None,"H4"]
    heading_stack: list[str] = []

    # 将md_content按行进行分割
    document_lines = md_content.splitlines()

    # 定义匹配标题的正则表达式
    title_reg = re.compile(r"^#{1,6}\s.+")

    # 对所有行进行遍历
    for line in document_lines:
        # 去除行前后空格
        line_strip = line.strip()

        if not line_strip:
            continue

        # 判断是不是代码标记行
        if line_strip.startswith("```") or line_strip.startswith("~~~"):
            is_code = not is_code  # 状态取反  分别对应的是进入/跳出两种状态
            if not current_title:
                orphan_lines.append(line_strip)
            else:
                current_content_lines.append(line_strip)
            continue

        # 判断是否处于代码块内部
        if is_code:
            if not current_title:
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)
            continue

        # 判断是不是标题
        if title_reg.match(line_strip):
            # 如果是标题 -- 标题的处理逻辑
            # 获取当前标题级别
            heading_level = len(line_strip) - len(line_strip.lstrip("#"))
            if current_title and len(current_content_lines) > 0:
                if not has_flushed_first_chunk and orphan_lines:
                    current_content_lines = orphan_lines + current_content_lines
                    has_flushed_first_chunk = True
                    orphan_lines = []
                full_content = f"{current_title}\n" + "\n".join(current_content_lines)
                chunks.append({
                    "title": current_title,
                    "content": full_content,
                    "file_title": file_title
                })
                # 清空正文缓冲区
                current_content_lines = []
            # 维护多级标题栈
            while len(heading_stack) < heading_level:
                heading_stack.append(None)

            heading_stack = heading_stack[:heading_level]

            heading_stack[heading_level - 1] = line_strip
            current_title = "_".join([h for h in heading_stack if h])
            continue
        else:
            # 如果不是标题 -- 正文的处理逻辑
            if not current_title:
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)

    if current_title and (len(current_content_lines) > 0 or (not has_flushed_first_chunk and len(orphan_lines) > 0)):
        if not has_flushed_first_chunk and len(orphan_lines):
            current_content_lines = orphan_lines + current_content_lines
            has_flushed_first_chunk = True
            orphan_lines = []
        full_content = f"{current_title}\n" + "\n".join(current_content_lines)
        chunks.append({
            "title": current_title,
            "content": full_content,
            "file_title": file_title
        })
    elif not current_title and len(orphan_lines):
        chunks.append({
            "title": file_title,
            "content": "\n".join(orphan_lines),
            "file_title": file_title
        })
    logger.info(f"已经根据标题进行多级切块（极简稳健版），现有的块: {len(chunks)}")
    return chunks


def _split_chunk_content(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    sub_chunk: list[dict[str, Any]] = []
    content = chunk.get('content')
    prefix = chunk.get('title') + '\n'
    deal_content = content[len(prefix):]
    rc_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；"],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,

    )
    for index, text in enumerate(rc_splitter.split_text(deal_content), 1):
        sub_chunk.append({
            'file_title': chunk.get('file_title'),
            'parent_title': chunk.get('title'),
            'title': f'{chunk.get('title')}_{index}',
            'part': index,
            'content': prefix + text,
        })
    return sub_chunk


def _merge_chunk_content(refine_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    合并同一个主题下，文本内容小于400，合并后小于1000的chunk
    :param refine_chunks:
    """
    merge_chunks: list[dict[str, Any]] = []
    base_chunk: dict[str, Any] = None
    for next_chunk in refine_chunks:
        if base_chunk is None:
            base_chunk = next_chunk
            continue
        # 如果 文本内容大于400
        too_long = len(next_chunk['content']) > CHUNK_SIZE
        if not too_long:
            # 是否是同一主题
            is_same_parent_title = (base_chunk.get("parent_title") and base_chunk.get("parent_title") == next_chunk.get(
                "parent_title"))
            if is_same_parent_title:
                # 合并是否超过1000
                base_content: str = base_chunk.get("content")
                next_cleared_content: str = next_chunk.get("content")[len(next_chunk.get("parent_title")) + 1:]
                merged_too_long = (len(base_content) + len(next_cleared_content)) > CHUNK_MAX_SIZE
                if not merged_too_long:
                    base_chunk["content"] = f'{base_content}\n{next_cleared_content}'
                else:
                    merge_chunks.append(base_chunk)
                    base_chunk = next_chunk
            else:
                merge_chunks.append(base_chunk)
                base_chunk = next_chunk

        else:
            merge_chunks.append(base_chunk)
            base_chunk = next_chunk
    if base_chunk:
        merge_chunks.append(base_chunk)
    logger.info(f"完成小于400的chunk的合并,合并后的数量:{len(merge_chunks)}")
    return merge_chunks


@step_log("step_3_refine_split_and_merge_chunks")
def step_3_refine_split_and_merge_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    进行chunks的精细切割(切+和)
    :param chunks: 经过语义切分后的chunks
    """
    # 1.定义一个接收本方法返回的chunk 列表
    refine_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        if len(chunk['content']) > CHUNK_SIZE:
            # 精细切割
            refine_chunks.extend(_split_chunk_content(chunk))
        else:
            refine_chunks.append(chunk)
    logger.info(f"chunks经过超长以后向短切割处理! 切割后的数量:{len(refine_chunks)}")
    # 短文本合并处理
    # 触达400[尽量再次处理 -> 1000] 相邻合并大于600的场景!
    refine_chunks = _merge_chunk_content(refine_chunks)

    return refine_chunks


@step_log("step_4_padding_chunks_metadata")
def step_4_padding_chunks_metadata(chunks: list[dict[str, Any]]):
    for chunk in chunks:
        if 'parent_title' not in chunk:
            chunk['parent_title'] = chunk.get('title')
        if 'part' not in chunk:
            chunk['part'] = 1


@step_log("step_5_backup_chunks_json")
def step_5_backup_chunks_json(chunks: list[dict[str, Any]], md_path):
    json_path = Path(md_path).parent / f'{Path(md_path).stem}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    logger.info(f"完成chunks数据的备份,备份位置:{str(json_path)}")


@node_log('node_document_split')
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点：文档切分节点
    """
    add_running_task(state.get('task_id'), 'node_document_split')
    # =================== step1 校验并获取参数 ===================
    md_content, file_title, md_path = step_1_validate_get_data(state)
    # =================== step2 根据语义文本切分 ===================
    # 根据标题进行内容切分
    chunks: list[dict[str, Any]] = step_2_split_document_by_title(md_content, file_title)

    # =================== step3 进行chunks的精细切割(切+和) ===================
    chunks: list[dict[str, Any]] = step_3_refine_split_and_merge_chunks(chunks)

    # =================== step4 补充parent_tile、part属性 ===================
    step_4_padding_chunks_metadata(chunks)

    # =================== step5 备份chunks，更新state ===================
    step_5_backup_chunks_json(chunks, md_path)
    state['chunks'] = chunks
    add_done_task(state.get('task_id'), 'node_document_split')
    return state


# def test():
#     file_path = r'D:\Python project\Shopkeeper_Brain\output\hak180产品安全手册\hak180产品安全手册_new.md'
#     document = Path(file_path).read_text(encoding='utf-8')
#     chunks = step_2_split_document_by_title(document, 'hak180产品安全手册')
#     print(f'总共{len(chunks)}个 chunk')
#     for chunk in chunks:
#         print(chunk)


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from utils.path_util import PROJECT_ROOT

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册_new.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        # result_state = node_md_img(test_state)
        # logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(test_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk")
        for chunk in final_chunks:
            print(chunk)
            print('=================================================================')
