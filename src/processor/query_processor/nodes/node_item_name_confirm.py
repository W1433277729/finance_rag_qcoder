"""
    @Desc   :
    @Time   :2026/9/14 16:47
    @Author :爱吃肯德基
"""
import math
import sys
import time
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv, find_dotenv

from src.common.config.milvus_config import milvus_config
from src.processor.query_processor.state import QueryGraphState
from src.utils.load_prompt import load_prompt
from src.utils.task_utils import add_running_task, add_done_task
from src.utils.clients.mongo_history_utils import get_recent_messages, save_chat_message
from src.utils.lm.lm_utils import get_llm_client
from src.utils.lm.embedding_utils import generate_embeddings
from src.utils.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from src.common.logging.logger import logger, node_log, step_log

load_dotenv(find_dotenv())

# 主体名称确认阈值：高于该分数 → 直接确认 [0.75]
ITEM_NAME_CONFIRM_THRESHOLD = 0.70
# 主体名称候选阈值：介于两者之间 → 让用户选择
ITEM_NAME_CANDIDATE_THRESHOLD = 0.60
# 给用户选择时，最多展示几个候选
ITEM_NAME_OPTIONS_TOPK = 2
ITEM_NAME_CONFIRMED_TOPK = 1


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: QueryGraphState) -> tuple[str, str]:
    """
     获取必要参数和检验
    :param state:
    :return:
    """
    # 1.获取请求参数
    session_id = state.get("session_id")
    original_query = state.get("original_query")
    # 2.非空校验
    if (not session_id) or (not original_query):
        logger.error(f"session_id或者original_query为空,业务无法继续,提前终止!")
        raise ValueError(f"session_id或者original_query为空,业务无法继续,提前终止!")
    # 3.返结果
    return original_query, session_id


@step_log("step_2_get_history_by_session_id")
def step_2_get_history_by_session_id(session_id: str) -> list[dict]:
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
    logger.info(f"根据:{session_id}查询到的聊天记录:{len(history_list)}")
    # 2.筛选有效的数据
    history_list = [history for history in history_list if history.get("item_names")]
    logger.info(f"根据:{session_id}过滤后的有效聊天记录:{len(history_list)}")
    # 3.返回聊天记录结果
    return history_list


@step_log("step_3_call_llm_rewritten_and_extract_itemnames")
def step_3_call_llm_rewritten_and_extract_itemnames(history_list: list[dict], original_query: str) -> dict[str, Any]:
    """
      重写问题 + 提取item_names
    :param history_list: 有效的聊天记录
    :param original_query: 原始问题
    :return: {item_names:[],rewritten_query:str}
    """
    # 1. 获取大语言模型客户端对象
    llm_model = get_llm_client(json_mode=True)
    # 2. 拼接提示词
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
     问题: 不要将回答全部放入history_text 会稀释规则 / 原始问题 / json参考格式
       km3 = 角色,原始问题->重写了什么样?关联的item_name -> user [很重要]
       km4 = 助手,重写问题->什么样的回答[:50]?关联的item_name -> assistant
     问题: 建议用户当前的问题,向上提取,放在第一行! 规则/json格式一并向上!
     问题: 强调,以本次用户的问题为主,进行提取! 用户可能转变提问的主体!!

    """
    history_text: str = None
    if history_list:
        history_text_list: list[str] = []

        for index, history in enumerate(history_list, start=1):
            if history.get("role") == 'user':
                # 编号:xxx,用户提问,原始问题:xx,重写后的问题:xxx，关联item_name：xxx
                history_text_list.append((
                    f"编号:{index},用户提问,原始问题:{history.get('text')},重写后的问题:{history.get('rewritten_query')},"
                    f"关联item_name:{','.join(history.get('item_names'))}"))
            else:
                # "编号:xxx,助手回答,重写问题:xx,回答结果:xx,关联item_name:xxx"
                history_text_list.append((
                    f"编号:{index},助手回答,重写问题:{history.get('rewritten_query')},回答结果:{history.get('text')[:50]},"
                    f"关联item_name:{','.join(history.get('item_names'))}"))
        history_text = "\n".join(history_text_list)
    else:
        history_text = "没有有效的历史聊天记录!"
    query = original_query

    prompt_text: str = load_prompt("rewritten_query_and_itemnames", query=query, history_text=history_text)

    message = HumanMessage(
        content=prompt_text
    )
    # 3. 封装调用链
    chains = llm_model | JsonOutputParser()
    # 4. 执行调用并获取结果
    json_dict: dict = chains.invoke([message])
    # 5. 处理和解析
    if "item_names" not in json_dict:
        json_dict['item_names'] = []
    if "rewritten_query" not in json_dict:
        json_dict['rewritten_query'] = original_query
    # 6. 返回结果
    return json_dict


@step_log("step_4_select_item_names_milvus")
def step_4_select_item_names_milvus(item_names: list[str]):
    """
     查询模型提供的item_name对应的向量数据的真实item_name
    :param item_names:
    :return: "{"llm_item_name":[]}"
    """
    milvus_search_result: dict[str, list[dict]] = {}
    # 0. 先批量生成向量,然后再循环获取向量
    result = generate_embeddings(item_names)
    # result {"dense":[[],[]],sparse:[{},{}]}
    # 1. 循环处理item_names获取每个模型提供的item_name进行查询
    for index, llm_item_name in enumerate(item_names):
        # 2. 使用嵌入式模型生成稠密和稀疏向量
        dense_vector = result['dense'][index]
        sparse_vector = result['sparse'][index]

        # 3. 稠密和稀疏向量转成对AnnSearchRequest
        reqs_list = create_hybrid_search_requests(dense_vector=dense_vector, sparse_vector=sparse_vector, limit=10)
        # 4. 启动混合查询 [[]]
        milvus_client = get_milvus_client()

        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs_list,
            ranker_weights=(0.8, 0.2),
            norm_score=True,
            limit=5,
            output_fields=['item_name']
        )
        # 5. 解析结果
        # response = [ [ {id / pk:x, distance: x , entity:{item_name}} , {} ,{}  ] ]
        # {
        #    llm_item_name: [{item_name:x,score:distance}]
        # }
        real_response = response[0]  # [ {id / pk:x, distance: x , entity:{item_name}} , {} ,{}  ]
        print(f'{real_response}')
        if not real_response:
            logger.warning(
                f"根据item_name:{llm_item_name}进行向量数据库的相似度搜索,值为空,向量数据库为空,跳出所有查询!")
            break

        llm_item_result_list: list[dict] = []
        # real_response =  [ {id / pk:x, distance: x , entity:{item_name}} , {} ,{}  ]
        for item in real_response:
            # item_name = {id / pk:x, distance: x , entity:{item_name}}
            llm_item_result_list.append(
                {
                    "item_name": item.get("entity", {}).get("item_name"),
                    "score": item.get("distance", 0.0)
                }
            )
        # [{item_name:x,score:x} , {}]

        # {llm_item_name 模型提供的名 : [{item_name:x,score:x}] }
        milvus_search_result[llm_item_name] = llm_item_result_list
    return milvus_search_result


@step_log("step_5_select_confirmed_and_option_item_names")
def step_5_select_confirmed_and_option_item_names(search_result: dict[str, list[dict]]):
    """
      从向量库查询的数据中提取可选和确认的列表
    :param search_result:
    :return:
    """

    # key = confirmed  key=option
    confirmed_list: list[str] = []  # 确认的item_name
    option_list: list[str] = []  # 可选的item_name
    for item_name, entry_list in search_result.items():
        high_list = [entry.get('item_name') for entry in entry_list if
                     entry.get('score', 0.0) >= ITEM_NAME_CONFIRM_THRESHOLD]
        middle_list = [entry.get('item_name') for entry in entry_list if
                       ITEM_NAME_CANDIDATE_THRESHOLD <= entry.get('score', 0.0) <= ITEM_NAME_CONFIRM_THRESHOLD]
        if high_list:
            confirmed_list.extend(high_list[:ITEM_NAME_CONFIRMED_TOPK])
            logger.info(f"{item_name}已经获取到可信的主体:{high_list[:ITEM_NAME_CONFIRMED_TOPK]}")
            continue
        if middle_list:
            option_list.extend(middle_list[:ITEM_NAME_OPTIONS_TOPK])
            logger.info(f"{item_name}没有可信的主体,但是有可选的主体:{middle_list[:ITEM_NAME_OPTIONS_TOPK]}")

    return {
        "confirmed": confirmed_list,  # expr item_name in [名字,名字]
        "option": option_list
    }


@step_log("step_6_change_state_property")
def step_6_change_state_property(state: QueryGraphState, rewritten_query, confirmed_option_dict):
    """
    更新state
      场景1: 有可信主体 -> item_names + rewritten_query,继续检索(不写answer)
      场景2: 无可信但有可选主体 -> 写answer反问用户澄清
      场景3: 无可信也无可选(知识/概念类问题,或产品未匹配) -> item_names置空,
             仅保留rewritten_query,走「无主体过滤」的全库检索(不写answer)
    :param state:
    :param rewritten_query:
    :param confirmed_option_dict:
    :return:
    """
    confirmed_list = confirmed_option_dict.get("confirmed", [])
    option_list = confirmed_option_dict.get("option", [])

    # 无论哪种场景,都保留改写后的问题供后续检索使用(空则回退原始问题)
    state['rewritten_query'] = rewritten_query or state.get('original_query')

    if confirmed_list:
        # 场景1: 有可信主体
        state['item_names'] = confirmed_list
        logger.info(f"已确认金融产品主体:{confirmed_list},更新item_names与rewritten_query!")
        return

    if option_list:
        # 场景2: 无可信,有可选 -> 反问用户澄清
        state['answer'] = f"您想查询的是以下哪个产品：{'、'.join(option_list)}？请明确后再提问。"
        logger.info(f"存在多个候选主体:{option_list},已写入反问answer!")
        return

    # 场景3: 无可信也无可选 -> 知识/概念类或未匹配查询,走无过滤全库检索
    state['item_names'] = []
    logger.info("未确认到具体金融产品主体,按知识/概念类问题继续走全库检索!")



@step_log("step_7_save_user_chat_message")
def step_7_save_user_chat_message(state: QueryGraphState):
    """
    保存用户聊天记录
    :param state:
    :return:
    """
    save_chat_message(
        session_id=state.get("session_id"),
        role="user",
        text=state.get("original_query"),  # 前端显示的就是这个字段
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names", []),
        image_urls=[]
    )

@node_log("node_item_name_confirm")
def node_item_name_confirm(state: QueryGraphState):
    #  1. 日志和任务处理 is_stream
    # 记录任务开始
    add_running_task(state["session_id"], 'node_item_name_confirm', state["is_stream"])
    # 1. 获取参数并且校验 original_query / session_id
    original_query, session_id = step_1_validate_and_get_data(state)
    # 2. 获取[有效的]历史聊天记录 session_id -> history_list
    history_list: list[dict] = step_2_get_history_by_session_id(session_id)
    # 3. 调用模型进行问题重写和item_name提取
    llm_dict: dict[str, Any] = step_3_call_llm_rewritten_and_extract_itemnames(history_list, original_query)
    confirmed_option_dict = {}
    if llm_dict.get("item_names"):
        # 4. 根据模型提取的item_name进行向量数据库的检索 [混合检索]
        search_result: dict[str, list[dict]] = step_4_select_item_names_milvus(llm_dict.get("item_names"))
        # 5. 根据向量数据库检索的结果(分)评判出两个列表 确认列表 / 可选列表
        # 模型1 华为60智能手机 -> item_name 华为meta60 + 分   item_name 华为meta40 + 分    item_name + 分
        # 模型2 苹果手机      -> item_name 苹果16 pro + 分    item_name + 分    item_name + 分
        #
        # 分数的区间  可信的分数区间 > 分 选1   || 可选的分数区间  > 可选 < 可信  选多个
        # 确定的列表 -> 在某个分值之上! 每一项分最高的! / 可选的列表 分最高的几个 top3
        confirmed_option_dict: dict[str, list[str]] = step_5_select_confirmed_and_option_item_names(search_result)
    # 6. 根据确认和可选列表存在性修改state
    # state -> item_names 可信确定 rewritten_query || 可选 answer || 没有可选  answer
    step_6_change_state_property(state, llm_dict.get("rewritten_query"), confirmed_option_dict)
    # #  7. 写入历史聊天记录(用户提问)
    step_7_save_user_chat_message(state)

    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return state


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "HAK 180 烫金机怎么用？",
        "is_stream": False,
    }
    result_state = node_item_name_confirm(mock_state)
    print(result_state)
