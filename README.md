# LangGraph 学习项目

基于 LangGraph 的渐进式学习项目，包含 7 节教程课程和一个完整的 AI 招聘助手实战应用。

## 项目结构

```
├── lessons/                    # 渐进式教程（7 课）
│   ├── lesson_01_core_concepts.py        # 核心概念：State、Node、Edge
│   ├── lesson_02_simple_chatbot.py       # 简单聊天机器人
│   ├── lesson_03_conditional_edges.py    # 条件边与动态路由
│   ├── lesson_04_react_agent.py          # ReAct 智能体
│   ├── lesson_05_checkpointing.py        # 状态持久化与回放
│   ├── lesson_06_human_in_the_loop.py    # 人工审核（HITL）
│   ├── lesson_07_multi_agent.py          # 多智能体协作
│   └── langgraph_slides.html             # 配套演示文稿
│
├── recruitment_assistant/      # 实战项目：AI 招聘助手
│   ├── graph/
│   │   ├── state.py            # RecruitmentState 状态定义
│   │   ├── agents.py           # 13 个节点函数（8 LLM + 3 Gate + 1 HITL Gate + 1 模板）
│   │   ├── tools.py            # LLM 工厂、JSON Mode、invoke_for_json 重试封装
│   │   └── pipeline.py         # StateGraph 构建、条件边、HITL 配置
│   ├── api/
│   │   ├── main.py             # FastAPI 应用入口
│   │   ├── models.py           # Pydantic 请求/响应模型
│   │   └── routers/
│   │       └── recruitment.py  # REST + SSE 接口
│   ├── frontend/
│   │   └── index.html          # 单文件前端（HTML + CSS + JS）
│   ├── prd.md                  # 产品需求文档
│   └── requirements.txt        # Python 依赖
│
└── docs/
    └── superpowers/specs/
        └── 2026-04-08-recruitment-assistant-design.md  # 设计文档
```

## 快速开始

### 环境准备

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r recruitment_assistant/requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 环境变量说明

| 变量 | 说明 | 示例 |
|------|------|------|
| `OPENAI_API_KEY` | LLM API 密钥 | `sk-...` |
| `DEFAULT_LLM_PROVIDER` | 模型名称 | `gpt-4o-mini` |
| `BASE_URL` | API 地址（可选，用于第三方兼容接口） | `https://api.example.com/v1` |

### 运行教程

```bash
# 按顺序逐课运行
python lessons/lesson_01_core_concepts.py
python lessons/lesson_02_simple_chatbot.py
# ...
```

### 运行招聘助手

```bash
# 启动 FastAPI 服务
uvicorn recruitment_assistant.api.main:app --reload --port 8000

# 浏览器打开前端页面
open recruitment_assistant/frontend/index.html
```

## 招聘助手架构

基于 LangGraph 的全流程招聘流水线，包含 13 个节点、4 个 HITL 暂停点：

```
START → parse_jd → resume_parser → screener → [HITL#1] → screening_gate
  → reject/need_more_info  → report_generator → END
  → phone_screen/onsite    → interview_generator → [HITL#2] → interview_gate
      → rejected → report_generator → END
      → approved → evaluator → [HITL#3] → collect_feedback
                 → merge_feedback → advisor → [HITL#4] → decision_gate
                    → offer    → offer_pack → report_generator → END
                    → rejected → report_generator → END
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/recruitment/start` | 提交 JD + 简历，启动流程 |
| `GET` | `/api/recruitment/{id}/state` | 查询当前状态 |
| `POST` | `/api/recruitment/{id}/resume` | HR 提交审核决定 |
| `POST` | `/api/recruitment/{id}/answers` | 录入面试回答 |
| `GET` | `/api/recruitment/{id}/report` | 获取最终报告 |

## 技术栈

- **AI 流水线**：LangGraph + LangChain
- **后端**：FastAPI + Uvicorn
- **前端**：HTML + Vanilla JS
- **LLM 可靠性**：JSON Mode (`response_format`) + `invoke_for_json` 自动重试 + `parse_json_from_llm` 多级容错
