# AGENTS.md

本文件为 AI 编码代理（及新加入的开发者）提供项目上下文。请在修改代码前通读。

## 1. 项目定位

**掌柜智库（Shopkeeper Brain）** 是一个基于 LangGraph 的 RAG（检索增强生成）系统，分为「文档导入」与「智能问答」两条工作流。

> 📌 **前提**：本项目同时是**练手/学习性质**的实战项目，改造过程中可优先保证功能跑通与思路清晰，不必过度追求生产级完备度。

- **原始实现域**：硬件产品说明书问答（示例数据曾为 `华为P60`、`HAK180 烫金机`、`万用表RS-12`）。整套流程（主体识别、消歧、答案生成、提示词）都围绕「**商品型号（item_name）**」这一核心实体构建。
- **目标改造域**：根据 `需求说明.md`（金融知识库项目需求），把系统改造成**面向普通用户的金融知识库查询系统**——支持金融产品、产品说明书、基金招募书、市场资讯、公司公告、政策解读、金融术语、FAQ、业务流程等资料的检索问答。

> ⚠️ 金融场景的核心特点：大量查询**没有明确产品实体**（如「什么是基金净值？」「基金赎回多久到账？」）。原流程强依赖 `item_name` 消歧，改造已加入「知识类查询旁路」来承接这类问题（见第 6 节改造进展）。

需求原文位于项目外部：`D:\学习资料\BJ20260615AI\17_尚硅谷大模型项目实战之掌柜智库实战\1.资料\金融\需求说明.md`

## 2. 技术栈

- Python >= 3.12，包管理用 **uv**（`pyproject.toml` + `uv.lock`，清华镜像源，仅 win32/AMD64）
- **LangGraph** 编排工作流，**LangChain** 封装 LLM 调用链
- **FastAPI** + uvicorn 提供 HTTP/SSE 接口
- **Milvus** 向量库：稠密（HNSW/COSINE）+ 稀疏（SPARSE_INVERTED_INDEX/IP）**混合检索**
- **BGE-M3** 生成稠密+稀疏向量；**bge-reranker-large** 重排序（本地模型，`BGE_DEVICE=cpu`）
- **Qwen（阿里百炼 / DashScope，OpenAI 兼容接口）** 作为 LLM 与 VL 模型
- **MongoDB** 存对话历史；**MinIO** 存图片；**MineRU** 做 PDF→Markdown
- **MCP WebSearch（DashScope）** 做联网搜索召回
- **RAGAS 0.4**（`eval/`）做 RAG 评估。注意：ragas 0.4.3 会 `import langchain_community.chat_models.vertexai`，而 langchain-community 0.4.x 已删除该模块，因此 `eval/judge.py` 顶部注入了桩模块（见该文件注释，勿删）
- 日志：**loguru**，统一从 `src.common.logging.logger` 导入

## 3. 目录结构与职责

```
src/
  api/
    file_import_service.py   # 导入服务，端口 8000：/upload /status/{task_id} /import.html
    query_service.py         # 查询服务，端口 8001：/query /stream/{sid} /history/{sid} /chat.html /health
  common/
    config/                  # 各类配置：milvus/embedding/lm/reranker/minio/mongo/bailian_mcp，dataclass 单例读 os.getenv
    logging/logger.py        # loguru 封装 + @node_log / @step_log 装饰器
    prompt/*.prompt          # 提示词模板，用 load_prompt(name, **kwargs) 加载
  page/                      # 前端 chat.html / import.html
  processor/
    import_processor/        # 导入工作流：main_graph.py(kb_import_app) + nodes/ + state.py
    query_processor/         # 查询工作流：main_graph.py(query_app) + nodes/ + state.py
  utils/
    clients/                 # milvus_utils / minio_utils / mongo_history_utils
    lm/                      # embedding_utils / lm_utils / reranker_utils
    load_prompt.py path_util.py sse_utils.py task_utils.py rate_limit_utils.py ...
doc/                         # 待导入资料（已替换为金融语料，按内容类型分子目录，见下）
tests/                       # pytest 测试
eval/                        # RAG 评估（与运行时逻辑隔离，只在进程内调用查询流程）
  kb_inventory.py            # 盘点 KB 实际入库的 file_title 与元数据分布
  build_dataset.py           # 合并 drafts → questions_v*.jsonl，校验并生成人工评审表
  judge.py                   # RAGAS 判官层（LLM 工厂 / 指标实例化 / 合规 rubric 加载）
  run_rag_eval.py            # 主入口：跑查询流程 → 规则断言 → 检索命中 → RAGAS 打分 → 报告
  report.py                  # 报告聚合与渲染
  clean_eval_sessions.py     # 清理 eval_ 前缀的 Mongo 会话历史（默认只预览）
  tracing.py                 # LangFuse 追踪预留接口（对应 --trace）
  dataset/                   # questions_v1.jsonl + drafts/ + review_questions_v1.md
  reports/                   # 报告 json/md 与运行日志（*.log 已 gitignore）
```

### doc/ 金融语料（已就位，对应需求 §4 内容导入范围）
原硬件 PDF（`hak180产品安全手册.pdf`、`万用表RS-12的使用.pdf`）已删除。现有资料按内容类型分目录，可直接映射到需求 §5 的 `content_type` 字段：

| 目录 | content_type | 文件 |
|------|-------------|------|
| `基金产品/` | 基金产品资料概要 | 华夏债券C、华夏国证自由现金流ETF联接、易方达智造优势混合C（均 PDF）|
| `上市公司年报/` | 公司公告/财报摘要 | 平安银行、招商银行、贵州茅台 2026 Q1 报告（PDF）|
| `基础知识/` | 金融术语/知识 | 基金基础知识（PDF）|
| `宏观经济&政策资料/` | 政策解读/市场资讯 | 2025国民经济统计公报、金融消费者权益保护实施办法、货币政策执行报告（PDF）|
| `用户FAQ/` | 常见问题 FAQ | ETF投教问答(.doc)、投资者问答(PDF)、消费者金融素养问卷(PDF)、私募基金投教问答(.docx)|
| `银行理财&风险揭示书/` | 产品风险揭示书 | 建行/中行/中银理财 风险揭示书（PDF）|

> ℹ️ **格式支持**：导入流程只支持 **PDF（MineRU 解析）和 MD**（见 `node_pdf_to_md` / `node_md_img`）。`用户FAQ/` 下的 `.doc` / `.docx`（ETF投教问答、私募基金投教问答）**已决定跳过**，当前 KB 未收录，故 FAQ 类问题只能落在 `投资者问答.pdf` 与 `消费者金融素养问卷调查报告.pdf` 上。

## 4. 两条工作流

### 导入流程 `kb_import_app`（import_processor/main_graph.py）
```
node_entry → (route_after_entry)
   ├─ is_md_read_enabled  → node_md_img
   └─ is_pdf_read_enabled → node_pdf_to_md → node_md_img
→ node_document_split → node_item_recognition → node_bge_embedding → node_import_milvus → END
```
- 文档切分：按 Markdown 标题多级切分 + `RecursiveCharacterTextSplitter` 精切（CHUNK_SIZE=600/MAX=1000/OVERLAP=50/MIN=400），短块合并；备份为同名 `.json`
- 主体识别 `node_item_recognition`：LLM 提取 `item_name`，写入 **kb_item_names** 集合
- 切片向量化后写入 **kb_chunks** 集合

### 查询流程 `query_app`（query_processor/main_graph.py）
```
node_item_name_confirm → (route_after_item_confirm)
   ├─ state['answer'] 已存在（消歧失败/反问）→ node_answer_output → END
   └─ 否则并发：node_search_embedding + node_search_embedding_hyde + node_web_search_mcp
→ node_rrf（RRF 融合，k=60）→ node_rerank → node_answer_output → END
```
- `node_item_name_confirm`：改写问题 + 提取 item_names → 向量库匹配真实主体。三种场景：① 有可信主体（≥0.70）→ 确认并继续检索；② 无可信但有候选（0.60~0.70）→ 写 `answer` 反问用户选择；③ 无可信也无可选 → **清空 item_names，走全库检索**（知识类旁路）
- 多路召回：普通向量 + HyDE；**联网搜索默认关闭**（`state['disable_web_search']` 默认 `True`，只走向量库检索，保证来源可溯源；需要时置 `False` 恢复 `node_web_search_mcp`）
- `node_answer_output`：拼 `answer_out.prompt`（结构化输出 + 合规护栏 + 无资料兜底）→ LLM（支持 SSE 流式 delta）→ 抽取图片 URL → 构建 `references` 引用来源 → 存历史

## 5. 编码约定（务必遵守）

- **绝对导入**：统一 `from src.xxx import ...`；`PROJECT_ROOT` 来自 `src.utils.path_util`（靠根目录 `.env` 定位）
- **节点文件命名**：`node_<name>.py`，导出函数 `node_<name>(state) -> state`
- **节点结构**：拆成多个 `step_N_xxx(state)` 私有步骤函数；节点入口做参数校验 → 业务 → 更新 state → `return state`
- **日志装饰器**：节点用 `@node_log("node_name")`，步骤用 `@step_log("step_N_xxx")`
- **任务追踪**：节点内首尾调用 `add_running_task(...)` / `add_done_task(...)`（导入用 `task_id`，查询用 `session_id` + `is_stream`）
- **State**：`TypedDict` + `create_default_state(**overrides)` / `create_query_default_state(**overrides)`（深拷贝默认值，避免全局污染）
- **配置**：新增配置放 `src/common/config/*_config.py`，用 `@dataclass` 单例 + `os.getenv`，环境变量同步补到 `.env.example`
- **提示词**：外置为 `.prompt` 文件，通过 `load_prompt` 加载，不要硬编码在 py 里
- **文件头**：保留 `@Desc / @Time / @Author` 注释块；文件末尾常带 `if __name__ == "__main__":` 单节点/全流程测试
- **Milvus 幂等**：入库前先按 `file_title == '...'` 删除旧数据再插入（注意 `==` 与字符串转义，见 `escape_milvus_string_utils.py`）
- **评估（`eval/`）**：走进程内 `query_app.invoke(...)`，不走 HTTP/SSE；每题独立 `session_id=eval_<run>_<题号>` 隔离历史（用完可用 `clean_eval_sessions.py` 清理）；判官固定 `temperature=0`、同一版题库，改完代码重跑同版对比；合规 rubric 外置在 `src/common/prompt/compliance_eval.prompt`
- **不提交 `.env`**（已 gitignore），只维护 `.env.example`

## 6. 金融域改造进展与遗留

> **改造决策（已与用户确认）**：① 范围＝分阶段，先做**最小问答闭环**；② `doc/用户FAQ` 的 `.doc/.docx` **跳过，只导入 PDF**；③ 金融元数据由 **LLM 自动抽取**（`finance_metadata_extract.prompt`，`content_type` 用固定枚举）。

### 已完成
1. **Schema 扩展**：`kb_chunks` 增加 `content_type/product_name/product_code/institution_name/risk_level/industry/market/publish_date/entry_name/source_file/source_path`；检索侧 `output_fields` 同步（`src/utils/clients/milvus_utils.py` 的 `CHUNK_OUTPUT_FIELDS`）。
2. **金融主体识别**：`node_item_recognition` 用 LLM 抽取 `item_name` 与金融元数据（上下文取前 7 个 chunk）；`item_name_recognition.prompt` / `product_recognition_system.prompt` 已改为金融主体识别。
3. **知识类旁路**：`node_item_name_confirm` 场景 ③ 清空 `item_names` 走全库检索，不再一律拒答/反问。
4. **结构化答案 + 合规护栏 + 兜底**：`answer_out.prompt` 输出【简要结论/主要内容/风险提示/注意事项】+【引用来源】，无资料时输出固定兜底话术「当前知识库中未检索到足够信息…」。
5. **引用来源回传**：查询响应新增 `references`（资料名/内容类型/产品名/机构/发布时间/来源文件）。
6. **导入页多文件排队**：`src/page/import.html` 支持多文件/文件夹勾选、串行排队（`MAX_CONCURRENCY=1`）、上传与处理逐文件真实进度 + 顶部总进度；`.doc/.docx` 标注「不支持」且默认不勾选；同名文件有「重名」提醒。
7. **磁盘层同名覆盖已修**：`/upload` 把上传文件存进 `output/<日期>/<task_id>/`，不同子目录同名文件不再互相覆盖。

### 遗留
1. **Milvus 层同名覆盖**：入库仍按 `file_title`（=文件名主干）先删后插，不同子目录同主名资料会互相覆盖；目前靠导入页的「重名」提醒规避，未彻底修复。
2. **评估指标**：`answer_relevancy` 已接入（`eval/embeddings.py` 把本地 BGE-M3 包成 ragas embeddings）；`answer_correctness` 与 `factual_correctness` 语义重叠，默认关闭，需要时用 `--metrics` 显式指定。
3. **LangFuse 追踪**：仅留接入点（`eval/tracing.py` + `--trace`），未安装依赖。
4. **图片逻辑**（`node_md_img`、`step_6_extract_chunk_and_url_image`）在金融文档场景基本不产出图片，保留未删。
5. **部分节点 `__main__` 测试块**仍引用已删除的硬件 PDF，需要时替换为 `doc/` 下语料。
6. **规则断言的局限**：`must_have`/`must_not` 是字面匹配（数字已支持千分位与单位换算），识别不了否定式表达；带否定语义的合规判断依赖合规 rubric 的 LLM 评分。
7. **报表/表格类切片召回不到（2026-09-22 评估定位）**：年报里的财务报表被切成原始 HTML 表格堆（`<td>`/`colspan=`），交叉编码器给这类切片极低分（含「合并净利润 38,048百万元」的切片只 0.117，排最后），断崖截断后不会进上下文 —— 于是「一季度净利润多少」这类问题会漏项或误取母公司口径（33,769）。实测放宽候选池到 limit=10 无效（该切片进池但仍排末位）、调大 `RERANK_MIN_TOPK` 也无用（徒增噪声）。根治要改**导入侧**：把 HTML 表格清洗成结构化文本（每行「指标名+本期/上期/同比」）、让单个指标行可被独立检索，代价是需要重新导入年报语料。**当前决定：暂不修**，题库中 `report-04` 已标记 `known_issue` 保留作对照。

### 评估基线（2026-09-22，43 题，判官 qwen-flash，联网搜索关闭）
检索标准来源命中率 37/41（90%）｜上下文精确率 0.84｜上下文召回率 0.85｜忠实度 0.70｜事实正确性 0.49｜合规 rubric 0.94（5 条归一化）。报告见 `eval/reports/`（题库 `eval/dataset/questions_v1.jsonl`）。

## 7. 运行方式

```bash
# 安装依赖
uv sync

# 配置环境变量：复制 .env.example 为 .env 并填写（LLM key、Milvus/Mongo/MinIO 地址、模型路径等）

# 启动导入服务（端口 8000）
uv run python -m src.api.file_import_service      # 或 uv run python src/api/file_import_service.py

# 启动查询服务（端口 8001）
uv run python -m src.api.query_service

# 单独调试某条工作流 / 某个节点：直接运行对应文件（每个文件带 __main__ 测试块）
uv run python -m src.processor.query_processor.main_graph
uv run python -m src.processor.import_processor.main_graph

# 测试
uv run pytest

# RAG 评估（题库见 eval/dataset/questions_v1.jsonl，评测时联网搜索自动关闭）
uv run python -m eval.kb_inventory                            # 盘点 KB 实际入库内容
uv run python -m eval.build_dataset --from-jsonl              # 校验题库并重建人工评审表
uv run python -m eval.run_rag_eval --limit 3 --metrics none    # 冒烟：只跑规则断言 + 检索命中
uv run python -m eval.run_rag_eval --tag full_v1               # 全量：含 RAGAS 判官打分
uv run python -m eval.clean_eval_sessions                      # 预览评估会话历史，加 --yes 执行删除
```

**依赖的外部服务**：Milvus(19530)、MongoDB(27017)、MinIO(9000)、可访问的 DashScope/百炼 API、本地已下载的 BGE-M3 与 bge-reranker-large 模型。

## 8. 提交信息约定

近期提交遵循 `feat(master):中文简述` 格式（如 `feat(master):多路召回、rrf排序完成`）。请沿用该风格，一次提交聚焦一个功能点。

## 9. 其它约定

- `.gitignore` 已忽略 `/output/`、`/doc/`、`*.log`、`/eval/reports/`、`/eval/dataset/drafts/`——`doc/` 下的金融语料 PDF **不纳入 git**，属本地资料。
- 环境变量文件 `.env` 已存在且配置完整（勿提交，勿打印其中密钥）。
- **评估相关的提交边界**：提交 `eval/*.py` 与冻结题库 `eval/dataset/questions_v1.jsonl`（+可选的评审表 md）；**不提交** `eval/reports/`（报告含逐题答案与语料片段原文，且可重新生成）与 `eval/dataset/drafts/`（出题草稿，已被冻结题库取代）。评估基线数字记在第 6 节，作为版本对比依据。
