"""
    @Desc   :
    @Time   :2026/9/14 16:47
    @Author :爱吃肯德基
"""
import math
import re
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
# 反问前的字面关联阈值：候选主体与模型提取的主体，中文字符重合率低于该值则不反问
# 目的：向量相似度落在候选区间 ≠ 用户问的就是库里这个产品。用户问的东西库里根本没有时
# （如「建行利得盈2013年第17期」），应走全库检索 + 无资料兜底（需求 §6.3），
# 而不是把无关候选抛给用户，那会让人误以为该产品在库中。
ITEM_NAME_OPTION_LITERAL_OVERLAP = 0.5


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


def _literal_overlap(probe: str, text: str, threshold: float | None = None) -> bool:
    """
    判断 probe 的字符有多大比例出现在 text 中（默认阈值 0.5），用于三道护栏：
    1) 反问护栏：候选主体是否与模型提取的主体字面相关（probe=提取名，text=候选）
    2) 主体护栏：匹配到的主体是否真的出现在用户问题/历史里（probe=主体，text=问题+历史）
    3) 资料线索匹配：线索是否指向某份真实资料（probe=线索，text=资料名，阈值更严见 DOC_HINT_LITERAL_OVERLAP）
    优先按中文汉字比对（避免数字/字母造成虚假重合，如「第17期」与「07期」共享 0/7/期），
    双方都无中文时退化为非空白字符比对。
    :return: True 表示字面相关；False 表示无关（大概率是模型臆测）
    """
    threshold = ITEM_NAME_OPTION_LITERAL_OVERLAP if threshold is None else threshold

    def char_set(value: str, cjk_only: bool) -> set[str]:
        value = str(value or '')
        if cjk_only:
            return {ch for ch in value if '\u4e00' <= ch <= '\u9fff'}
        return {ch for ch in value if not ch.isspace()}

    probe_cjk, text_cjk = char_set(probe, True), char_set(text, True)
    probe_chars, text_chars = (probe_cjk, text_cjk) if (probe_cjk and text_cjk) \
        else (char_set(probe, False), char_set(text, False))
    if not probe_chars or not text_chars:
        return False
    return len(probe_chars & text_chars) / len(probe_chars) >= threshold


# 知识库资料清单进程内缓存：反问候选与资料线索匹配都用它
_KB_FILE_TITLES_CACHE: list[str] | None = None

# 反问澄清时最多列出几个候选
CLARIFY_TITLE_PREVIEW = 8

# 单条资料线索最多可匹配几份资料：超过说明线索太泛（如"报告""风险"），不用作过滤条件
DOC_HINT_MAX_MATCHES = 4
# 资料线索与资料名的字面重合阈值：比主体匹配更严 —— 短线索容易靠"公""报"这类常见字蒙中 0.5 的线
DOC_HINT_LITERAL_OVERLAP = 0.75

# 指向不明判定的**确定性**规则（不依赖 LLM 自评，避免误反问）：
# ① 指示代词必须**带着"产品/资料"这类对象名词**才算指向不明：
#    这样「这个产品」「这份公告」会反问，而「这个指标是什么意思」（指代对象就在本句内，如"夏普比率"）
#    「它跟踪得紧不紧」（对象已在句中）不会误反问
CLARIFY_PRONOUN_RE = re.compile(
    r'(这个|这份|这只|这几个|这几份|该)(产品|基金|理财|报告|公告|文件|资料|说明书|揭示书|年报|季报|概要|机构|公司|银行)'
    r'|上述|本文'
)
# ② 「某 + 名词」：指"某一具体但未指明的对象"
CLARIFY_INDEFINITE_RE = re.compile(r'某(一)?(公司|机构|行业|政策文件|产品|基金|银行|报告|公告|理财)')


def _need_clarify(original_query: str, item_names: list[str]) -> bool:
    """
    是否需要先反问用户澄清：问题指向某个具体对象、但没说清是哪一个。
    规则（对应用户确认的策略 A"有代词就反问"）：
      1) 模型已抽出具体主体 → 对象明确，不反问；
      2) 问题里出现「指示代词 + 对象名词」（这个产品/这份公告/该基金…），或「上述/本文」；
      3) 问题用「某 + 名词」指代未指定对象。
    """
    if item_names:
        return False
    text = original_query or ''
    if CLARIFY_PRONOUN_RE.search(text):
        return True
    return bool(CLARIFY_INDEFINITE_RE.search(text))


def _load_kb_file_titles() -> list[str]:
    """加载知识库实际入库的资料名（file_title）清单，进程内只查一次。"""
    global _KB_FILE_TITLES_CACHE
    if _KB_FILE_TITLES_CACHE is not None:
        return _KB_FILE_TITLES_CACHE
    titles: list[str] = []
    try:
        client = get_milvus_client()
        if client is not None:
            collection = milvus_config.chunks_collection
            client.load_collection(collection_name=collection)
            rows = client.query(collection_name=collection, filter='chunk_id >= 0',
                                output_fields=['file_title'], limit=16384)
            titles = sorted({row.get('file_title') for row in rows if row.get('file_title')})
    except Exception as e:
        logger.warning(f"读取知识库资料清单失败,本次跳过资料过滤与候选列举:{str(e)}")
    _KB_FILE_TITLES_CACHE = titles
    logger.info(f"知识库资料清单加载完成,共{len(titles)}份")
    return titles


def _match_doc_filters(file_hints: list[str], context_text: str) -> dict:
    """
    把模型抽取的「资料线索」匹配成真实 file_title 白名单（需求 §11.4：泛指某份资料的问题）。
    两道护栏：① 线索必须能在用户问题/历史里字面找到依据（防模型臆测）；② 必须能匹配到知识库中的资料。
    :return: {"file_titles": [...]}；无有效线索时返回空字典（退回全库检索）
    """
    titles = _load_kb_file_titles()
    if not titles:
        return {}
    matched: list[str] = []
    for hint in (file_hints or []):
        hint = str(hint or '').strip()
        if not hint:
            continue
        if not _literal_overlap(hint, context_text):
            logger.info(f"资料线索[{hint}]未在问题/历史中出现,判定为模型臆测并丢弃")
            continue
        hit_titles = [title for title in titles
                      if hint in title or _literal_overlap(hint, title, DOC_HINT_LITERAL_OVERLAP)]
        if len(hit_titles) > DOC_HINT_MAX_MATCHES:
            logger.info(f"资料线索[{hint}]过于宽泛(命中{len(hit_titles)}份资料),不用于收窄检索")
            continue
        for title in hit_titles:
            if title not in matched:
                matched.append(title)
    if matched:
        logger.info(f"资料线索命中知识库资料:{matched}")
        return {"file_titles": matched}
    return {}


def _build_clarify_answer() -> str:
    """指向不明（含指代词或未指定对象且历史无法消解）时的反问话术，附带知识库可选资料。"""
    titles = _load_kb_file_titles()
    preview = '、'.join(f'《{title}》' for title in titles[:CLARIFY_TITLE_PREVIEW])
    suffix = '等' if len(titles) > CLARIFY_TITLE_PREVIEW else ''
    return (f"您想了解的是哪一份资料或哪一个产品？请明确具体名称后我再为您查询。\n"
            f"当前知识库中包含：{preview}{suffix}。")


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
            # 候选必须与问题主体有明显字面关联，否则视为「库外问题」，交给全库检索走兜底
            related_list = [candidate for candidate in middle_list if _literal_overlap(item_name, candidate)]
            if related_list:
                option_list.extend(related_list[:ITEM_NAME_OPTIONS_TOPK])
                logger.info(f"{item_name}没有可信的主体,但是有可选的主体:{related_list[:ITEM_NAME_OPTIONS_TOPK]}")
            else:
                logger.info(f"{item_name}的候选与问题主体无明显字面关联,判定为知识库外问题,"
                            f"不反问用户,改走全库检索/无资料兜底;候选={middle_list}")

    return {
        "confirmed": confirmed_list,  # expr item_name in [名字,名字]
        "option": option_list
    }


@step_log("step_6_change_state_property")
def step_6_change_state_property(state: QueryGraphState, rewritten_query, confirmed_option_dict,
                                 history_text: str = "", file_hints: list[str] | None = None):
    """
    更新state
      场景1: 有可信主体 -> item_names + rewritten_query,继续检索(不写answer)
      场景2: 无可信但有可选主体 -> 写answer反问用户澄清
      场景3: 无可信也无可选 -> item_names置空,走全库检索;若能匹配到资料线索则按资料收窄(需求 §11.4)
    :param history_text: 历史对话文本，用于校验主体/资料线索是否真的在上下文里出现过
    :param file_hints: 模型抽取的资料线索（如"货币政策执行报告"）
    :return:
    """
    confirmed_list = confirmed_option_dict.get("confirmed", [])
    option_list = confirmed_option_dict.get("option", [])

    # 无论哪种场景,都保留改写后的问题供后续检索使用(空则回退原始问题)
    state['rewritten_query'] = rewritten_query or state.get('original_query')

    # 主体护栏：匹配到的主体必须真的出现在用户问题或历史里，否则视为模型臆测并丢弃。
    # 典型反例：「报告里说科技贷款、绿色贷款…」被抽成 ['私募基金','平安银行']，
    # 一旦按主体过滤检索，真正该查的《中国货币政策执行报告》反而不会被检索到。
    context_text = f"{state.get('original_query') or ''} {history_text or ''}"
    if confirmed_list or option_list:
        dropped = [name for name in confirmed_list + option_list if not _literal_overlap(name, context_text)]
        if dropped:
            logger.info(f"主体未出现在问题/历史中,判定为模型臆测并丢弃:{dropped}")
            confirmed_list = [name for name in confirmed_list if name not in dropped]
            option_list = [name for name in option_list if name not in dropped]

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

    # 场景3: 无可信也无可选 -> 知识/概念类或未匹配查询,走无主体过滤的检索
    # 若问题点名了某份资料（需求 §11.4 公告资讯类），则按 file_title 收窄检索范围
    state['item_names'] = []
    state['doc_filters'] = _match_doc_filters(file_hints or [], context_text)
    if state['doc_filters']:
        logger.info(f"未确认到主体,但匹配到资料线索,按资料收窄检索:{state['doc_filters']}")
    else:
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
    # history_text 用于校验主体/资料线索是否真的在上下文出现过（多轮追问时对象可能只在历史里）
    history_text = ' '.join(
        f"{item.get('text', '')} {item.get('rewritten_query', '')}" for item in history_list
    )

    # 3b. 指向不明（含指示代词 / 「某X」未指定对象）→ 先反问用户澄清，不进入检索。
    #     判定用确定性规则 _need_clarify（不依赖 LLM 的 needs_clarification 自评：
    #     实测它会把"什么是最大回撤""老百姓收入水平"这类通用问题误标为需澄清）。
    if _need_clarify(original_query, llm_dict.get("item_names") or []):
        logger.info(f"问题指向不明(含指示代词或未指定对象),先反问澄清;改写问题:{llm_dict.get('rewritten_query')}")
        state['rewritten_query'] = llm_dict.get("rewritten_query") or original_query
        state['item_names'] = []
        state['doc_filters'] = {}
        state['answer'] = _build_clarify_answer()
        step_7_save_user_chat_message(state)
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
        return state

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
    step_6_change_state_property(state, llm_dict.get("rewritten_query"), confirmed_option_dict, history_text,
                                 llm_dict.get("file_hints"))
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
