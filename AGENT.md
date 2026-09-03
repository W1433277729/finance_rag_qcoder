# AI 测试生成指令（Python + pytest）

## 项目介绍
掌柜智库：企业级私有知识库智能问答系统，可以通过上传pdf、markdown文件，将数据保存至向量数据库中。
- 主要部件：
  - import_processor:原始文档 → 结构化解析 → 语义切片 → 元数据提取 → 混合向量生成 → 向量数据库存储
  - query_processor:用户提问 → 意图理解 → 多路召回 → 结果融合 → 重排序 → LLM 生成答案 → SSE 流式输出

## 项目环境
- 源码目录：`src/`
- 测试目录：`tests/`
- 测试框架：pytest
- 覆盖率工具：pytest-cov
- 运行命令：`pytest --cov=src --cov-report=term --cov-report=html`

## 📁 目录结构
```text
Shopkeeper_Brain/
├── .env
├── doc/
├── logs/
├── pyproject.toml
├── src/
│   ├── api/    # api相关代码
│   ├── common/ # 公共配置、日志、提示词模版等
│   ├── page/   # web页面
│   ├── processor/  # 处理逻辑
│   │   ├── import_processor/   # 导入流程
│   │   └── query_processor/    # 问答流程
│   └── utils/  # 工具包
├── tests/  # 测试代码
└── uv.lock
```

## 开发进度
- 导入流程:
  - 整体框架已经搭好，具体节点实现逻辑还没有填充

## 测试生成硬性规则（AI 必须遵守）
1. **文件命名**：测试文件必须放在 `tests/` 下，与原模块的路径保持一致，命名为 `test_<对应模块名>.py`，例如`src/aaa.py`，`test/src/test_aaa.py`,
2. **函数命名**：测试函数以 `test_` 开头，命名格式 `test_<被测函数>_<测试场景>`（例如 `test_divide_by_zero`）。
3. **多组数据**：必须使用 `@pytest.mark.parametrize`，禁止写重复的测试函数。
4. **异常测试**：使用 `with pytest.raises(异常类型, match="错误信息")`。
5. **外部依赖（网络/DB/文件）**：必须使用 `monkeypatch` 或 `unittest.mock` 进行模拟，不允许真正发起请求或读写磁盘。
6. **共享数据**：跨文件的共用数据或初始化逻辑，放在 `tests/conftest.py` 中定义为 `@pytest.fixture`。
7. **覆盖率要求**：新代码行覆盖率需达到 **80% 以上**。

## 输出格式要求
- 只输出完整的 Python 代码，不要输出 Markdown 解释（除非我明确询问）。
- 生成的代码必须符合 PEP 8 规范。
- 如果被测函数有文档字符串，请基于文档描述设计测试。

## 特殊项目约定（如有，请修改）
- 异步代码请使用 `pytest-asyncio` + `@pytest.mark.asyncio`。
- 敏感配置请从环境变量读取，测试时使用 `monkeypatch.setenv`。
