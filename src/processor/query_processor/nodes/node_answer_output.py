"""
    @Desc   :
    @Time   :2026/9/14 16:50
    @Author :爱吃肯德基
"""
import re
import time
import sys
from src.common.logging.logger import logger, node_log, step_log
from src.processor.query_processor.state import QueryGraphState
from src.utils.clients.mongo_history_utils import get_recent_messages, save_chat_message
from src.utils.lm.lm_utils import get_llm_client
from src.utils.load_prompt import load_prompt
from src.utils.sse_utils import push_to_session, SSEEvent
from src.utils.task_utils import add_running_task, add_done_task


@step_log("step_1_answer_exists_in_state")
def step_1_answer_exists_in_state(state: QueryGraphState) -> bool:
    answer = state.get("answer")
    if answer:
        logger.info(f"第一个节点没有明确的item_name,已经有answer:{answer}")
        return True
    else:
        logger.info(f"没有answer,本次正常模型互动获取答案和图片即可!")
        return False


@step_log("step_2_validate_and_get_data")
def step_2_validate_and_get_data(state: QueryGraphState):
    reranked_docs = state.get("reranked_docs")
    session_id = state.get("session_id")
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")
    is_stream = state.get("is_stream", False)

    if (not reranked_docs) or (not session_id) or (not item_names) or (not rewritten_query):
        logger.error("核心参数为空,业务无法继续,提前终止!")
        raise ValueError("核心参数为空,业务无法继续,提前终止!")
    return reranked_docs, session_id, item_names, rewritten_query, is_stream


@step_log("step_3_get_history_by_session_id")
def step_3_get_history_by_session_id(session_id: str) -> list[dict]:
    """
      获取当前session_id对应有效的聊天记录
    :param session_id:
    :return:
    """
    # 1.调用mongo的函数获取聊天记录
    """
      {
        _id:ObjectId(1),
        role: user / assistant
        text: user-> original_query  assistant -> answer
        rewritten_query => 重写的问题
        item_names = [] 关联的item_name 不为空!!
        image_urls = []
        ts = 时间戳...
       }
    """
    history_list: list[dict] = get_recent_messages(session_id=session_id, limit=10)
    logger.debug(f"根据:{session_id}查询到的聊天记录:{len(history_list)}")
    # 2.筛选有效的数据
    history_list = [item for item in history_list if item.get("item_names")]
    logger.debug(f"根据:{session_id}过滤后的有效聊天记录:{len(history_list)}")
    # 3.返回聊天记录结果
    return history_list


@step_log("step_4_create_answer_prompt")
def step_4_create_answer_prompt(reranked_docs, history_message, item_names, rewritten_query):
    """
      拼接提示词
    :param reranked_docs:
    :param history:
    :param item_names:
    :param rewritten_query:
    :return:
    """
    # reranked_docs => [{ chunk_id , [  title , text , type , score ] ,url   }]
    context: str = ""
    for doc in reranked_docs:
        context += (f"标题:{doc.get('title')},数据类源:{'联网搜索' if doc.get('type') == 'web' else '向量数据库'}, "
                    f"置信度:{doc.get('score')},内容:{doc.get('text')} \n")

    # history_text:str = ""
    if history_message:
        history_text_list: list[str] = []

        for index, history in enumerate(history_message, start=1):
            if history.get("role") == 'user':
                history_text_list.append((
                    f"编号:{index},用户提问,原始问题:{history.get('text')},重写后的问题:{history.get('rewritten_query')},"
                    f"关联item_name:{','.join(history.get('item_names'))}"))
            else:
                history_text_list.append((
                    f"编号:{index},助手回答,重写问题:{history.get('rewritten_query')},回答结果:{history.get('text')[:50]},"
                    f"关联item_name:{','.join(history.get('item_names'))}"))
        history_text = "\n".join(history_text_list)
    else:
        history_text = "没有有效的历史聊天记录!"

    # 关联主体
    item_names_str: str = ",".join(item_names)

    answer_prompt_text = load_prompt("answer_out", context=context, history=history_text, item_names=item_names_str,
                                     question=rewritten_query)
    return answer_prompt_text


@step_log("step_5_call_llm_create_answer")
def step_5_call_llm_create_answer(answer_prompt: str, is_stream: bool, session_id: str) -> str:
    """
      调用模型回答问题! -> answer  流式需要stream执行模型,每次推送delta
        is_stream = True
           流式 -> 模型 -> 一段 一段 -> push_to_session(delta)
        answer 同步 / final
    :param answer_prompt:
    :param is_stream:
    :return:
    """
    # 1.获取大语言模型对象
    llm_model = get_llm_client()
    # 2.声明answer
    answer: str = ""
    # 3.根据流式状态判断执行
    if is_stream:
        stream = llm_model.stream(answer_prompt)
        for chunk in stream:
            chunk_str = chunk.content
            # 每块都要给前端返回 -> push_to_session(session_id,事件类型,数据字典) -> 队列 -> sse -> 推
            push_to_session(
                session_id=session_id,
                event=SSEEvent.DELTA,
                data={"delta": chunk_str}  # d.delta
            )
            answer += chunk_str
    else:
        # answer
        response = llm_model.invoke(answer_prompt)
        answer = response.content
    # 4.返回answer
    return answer


@step_log("step_6_extract_chunk_and_url_image")
def step_6_extract_chunk_and_url_image(reranked_docs) -> list[str]:
    """
      提取我们参考内容中的图片地址
    :param reranked_docs:
    :return:
    """
    """
      reranked_docs = [{text:"文本 ![](http..) xxx  ![](http..)" , url: "xx.jpg"}]
    """
    rep = re.compile(r"\!\[.*?\]\((.*?)\)")  # 匹配md中的图片  findall  http.. http..
    # (".png", ".jpg", ".gif", ".jpeg", ".svg")
    image_urls: list[str] = []
    # 循环获取每个chunk片段
    for chunk in reranked_docs:
        # 先判断路径是不是
        url: str = chunk.get("url")
        text = chunk.get("text")
        # 再判断内容有没有
        if url:
            if url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")):
                image_urls.append(url)
        if text:
            image_url_list: list[str] = rep.findall(text)
            if image_url_list:
                image_urls.extend(image_url_list)
    logger.info(f"完成切片中的图片提取,提取数量:{len(image_urls)},内容:{image_urls}")
    return image_urls


@step_log("step_7_save_assistant_message")
def step_7_save_assistant_message(state):
    """
      保存模型回答的聊天记录
    :param state:
    :return:
    """
    save_chat_message(
        session_id=state.get("session_id"),
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names", []),
        image_urls=state.get("image_urls", [])
    )


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState):
    """
    节点功能：进行过处理可以是流式输出可以整体输出！
    """
    logger.info("---node_answer_output 节点处理开始---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 1、判断answer是否存在
    has_answer = step_1_answer_exists_in_state(state)
    # 2、不存在，模型回答answer以及提取image_url的过程
    if not has_answer:
        # 2.1 获取参数及校验
        reranked_docs, session_id, item_names, rewritten_query, is_stream = step_2_validate_and_get_data(state)
        # 2.2 读取有效的聊天记录
        history = step_3_get_history_by_session_id(session_id)
        # 2.3 拼接提示词
        answer_prompt: str = step_4_create_answer_prompt(reranked_docs, history, item_names, rewritten_query)
        # 2.4 调用模型获取answer
        # 1.流式->push  2.流式|非流式 answer
        answer: str = step_5_call_llm_create_answer(answer_prompt, is_stream, session_id)
        # 2.5 提取片段中的图片image_urls
        image_urls: list[str] = step_6_extract_chunk_and_url_image(reranked_docs)
        # 2.6 更新state
        state['answer'] = answer
        state['image_urls'] = image_urls

    # 3、保存聊天记录
    step_7_save_assistant_message(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info("---node_answer_output 节点处理结束---")
    return state


if __name__ == "__main__":
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "type": "milvus",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)
            """,
        }
    ]
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？",
         "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法"},
    ]
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,
        "answer": None,
    }
    result = node_answer_output(mock_state)
    print(result)
