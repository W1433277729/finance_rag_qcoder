# AGENTS.md

本文件为 AI 编码代理（及新加入的开发者）提供项目上下文。请在修改代码前通读。

## 1. 项目定位

**掌柜智库（Shopkeeper Brain）** 是一个基于 LangGraph 的 RAG（检索增强生成）系统，分为「文档导入」与「智能问答」两条工作流。

> 📌 **前提**：本项目同时是**练手/学习性质**的实战项目，改造过程中可优先保证功能跑通与思路清晰，不必过度追求生产级完备度。

- **原始实现域**：硬件产品说明书问答（示例数据曾为 `华为P60`、`HAK180 烫金机`、`万用表RS-12`）。整套流程（主体识别、消歧、答案生成、提示词）都围绕「**商品型号（item_name）**」这一核心实体构建。
- **目标改造域**：根据 `需求说明.md`（金融知识库项目需求），把系统改造成**面向普通用户的金融知识库查询系统**——支持金融产品、产品说明书、基金招募书、市场资讯、公司公告、政策解读、金融术语、FAQ、业务流程等资料的检索问答。

> ⚠️ 改造的核心难点：金融场景包含大量**无明确产品实体**的查询（如「什么是基金净值？」「基金赎回多久到账？」），而现有流程强依赖 `item_name` 消歧，需重点调整（见第 6 节差距清单）。

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

> ⚠️ **格式支持缺口**：`用户FAQ/` 下有 `.doc`、`.docx` 文件，而现有导入流程只支持 **PDF（MineRU 解析）和 MD**（见 `node_pdf_to_md` / `node_md_img`）。改造时需决定是否：转换为 PDF/MD、或新增 Office 文档解析分支。

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
- `node_item_name_confirm`：改写问题 + 提取 item_names → 向量库匹配真实主体；阈值 `>=0.70` 直接确认，`0.60~0.70` 反问用户选择，否则拒答（写入 `answer`）
- 多路召回：普通向量、HyDE、联网搜索
- `node_answer_output`：拼 `answer_out.prompt` → LLM（支持 SSE 流式 delta）→ 抽取图片 URL → 存历史

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
- **不提交 `.env`**（已 gitignore），只维护 `.env.example`

## 6. 当前实现 vs 金融需求的差距清单（改造重点）

> **本轮改造决策（已与用户确认）**：① 范围＝分阶段，先做**最小问答闭环**；② `doc/用户FAQ` 的 `.doc/.docx` **本轮跳过，只导入 PDF**；③ 金融元数据（content_type/产品名/代码/风险等级等）由 **LLM 自动抽取**。

1. **数据 Schema 缺字段**：`kb_chunks` 现仅有 `file_title/item_name/title/parent_title/part/content`。需求 §5 要求补充：`content_type、product_name、product_code、institution_name、risk_level、industry、market、publish_date、entry_name、source_file、source_path`。需同步改 schema、切分/识别节点、检索 `output_fields`。
2. **item_name 语义**：现指「商品型号」，金融域应映射到「产品名称/产品代码/机构」。`product_recognition_system.prompt`（现为"你是商品识别专家"）、`item_name_recognition.prompt` 需改为金融主体识别。
3. **无产品实体的查询被拦截**：`node_item_name_confirm` 在无法确认 item_name 时直接写 `answer` 拒答/反问。金融知识、术语、FAQ、业务流程类问题（如「什么是净值型理财」）**没有具体产品**，需增加「知识/概念类查询」旁路，允许无 item_name 时正常走检索问答。
4. **答案结构**：`answer_out.prompt` 现为产品说明书 + 图片导向。需求 §6 要求结构化输出（简要结论/主要内容/风险提示/注意事项/引用来源），§6.2 引用来源，§6.3 无资料时的固定兜底话术。
5. **合规护栏缺失**：需求 §7 要求——不提供投资建议、不承诺收益（禁用"保本/稳赚不赔"等）、风险提示谨慎、不保证实时性。需在提示词与后处理中加入合规约束。
6. **引用来源未回传**：查询响应目前只返回 `answer` + `image_urls`，需增加来源信息字段（资料名称/内容类型/产品名/发布时间/来源文件）。
7. **图片相关逻辑**（`node_md_img`、`step_6_extract_chunk_and_url_image`）在金融文档场景可能弱化，按需保留。
8. **导入格式仅支持 PDF/MD**：`doc/用户FAQ/` 含 `.doc/.docx`，现流程无法解析；且 `content_type` 等金融元数据目前无从自动填充（可考虑按 `doc/` 子目录名推断，或在导入接口/识别节点中补全）。
9. **失效的测试入口**：`import_processor/main_graph.py`、`node_document_split.py` 等文件的 `if __name__ == "__main__":` 测试块仍硬编码引用已删除的 `hak180产品安全手册.pdf` / `万用表RS-12的使用.pdf`，改造后需替换为 `doc/` 下的金融语料。

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
```

**依赖的外部服务**：Milvus(19530)、MongoDB(27017)、MinIO(9000)、可访问的 DashScope/百炼 API、本地已下载的 BGE-M3 与 bge-reranker-large 模型。

## 8. 提交信息约定

近期提交遵循 `feat(master):中文简述` 格式（如 `feat(master):多路召回、rrf排序完成`）。请沿用该风格，一次提交聚焦一个功能点。

## 9. 其它约定

- `.gitignore` 已忽略 `/output/` 与 `/doc/`——`doc/` 下的金融语料 PDF **不纳入 git**，属本地资料。
- 环境变量文件 `.env` 已存在且配置完整（勿提交，勿打印其中密钥）。
