# 招聘助手分析系统 设计文档

**日期**：2026-04-08
**状态**：已批准
**作者**：CodeMaker + 用户

---

## 1. 项目概述

基于 LangGraph 构建一个全流程招聘助手，支持简历解析、筛选评分、面试题生成、面试评估和录用建议，并在关键节点引入 HR 人工审核（Human-in-the-Loop）。对外通过 FastAPI 提供 REST API，前后端分离。

### 1.1 核心目标

- 自动化招聘流水线的重复性 AI 分析工作
- 在关键决策点保留 HR 的人工控制权
- 为 LangGraph 高级模式（多智能体 + HITL + Checkpointing）提供完整的实战示例

### 1.2 技术栈

| 层 | 技术 |
|----|------|
| AI 流水线 | LangGraph + LangChain |
| LLM | 通过 `DEFAULT_LLM_PROVIDER` + `BASE_URL` 配置 |
| 后端 API | FastAPI + Uvicorn |
| 持久化 | LangGraph MemorySaver（开发） / SqliteSaver（生产） |
| 前端 | HTML + Vanilla JS（单文件，后续可替换） |

---

## 2. 项目结构

```
recruitment_assistant/
├── graph/
│   ├── __init__.py
│   ├── state.py          # RecruitmentState 定义
│   ├── agents.py         # 13 个节点函数（对齐 PRD 8 节点架构）：
    │                        #   8 个 LLM 节点：parse_jd, resume_parser, screener,
    │                        #     interview_generator, evaluator, merge_feedback, advisor, offer_pack
    │                        #   3 个 gate 门节点：screening_gate, interview_gate, decision_gate
    │                        #   1 个 HITL 门：collect_feedback
    │                        #   1 个模板拼接：report_generator
│   ├── tools.py          # 辅助工具函数（LLM 工厂、JSON Mode、invoke_for_json 重试封装）
│   └── pipeline.py       # 图构建、条件边、HITL 配置、compile()
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI 应用入口 + CORS
│   ├── models.py         # Pydantic 请求/响应模型（见第 5.3 节）
│   └── routers/
│       ├── __init__.py
│       ├── recruitment.py # 招聘流程路由
│       └── session.py     # 会话查询路由
├── frontend/
│   └── index.html        # 单文件前端（HTML + CSS + JS）
├── .env                  # 环境变量（已有）
└── requirements.txt      # 新增依赖：fastapi, uvicorn
```

---

## 3. 核心数据模型

### 3.1 RecruitmentState（LangGraph 状态）

```python
class RecruitmentState(TypedDict):
    # ---- 输入 ----
    session_id: str           # 会话唯一标识（UUID v4）
    jd_text: str              # 岗位要求（Job Description）原文
    resume_text: str          # 简历原文

    # ---- JD 解析（PRD: parse_jd）----
    parsed_jd: dict           # 结构化 JD：
                              #   {title, required_skills, nice_to_have_skills,
                              #    experience_years, education, responsibilities}

    # ---- 简历解析 ----
    parsed_resume: dict       # 结构化简历：{name, phone, email, skills,
                              #   experience, education, summary}

    # ---- 筛选评分（PRD: screen_resume）----
    score: int                # 匹配分数 0-100
    score_reason: str         # 评分总结理由
    evidence: list            # 逐项证据引用：
                              #   [{"requirement": str, "matched": bool,
                              #     "citation": str, "comment": str}]

    # ---- 筛选分流（PRD: route_screen）----
    screen_route: str         # 分流结果：reject / phone_screen / onsite / need_more_info

    # ---- 面试题生成（PRD: generate_questions，含 rubrics）----
    interview_questions: list  # 面试题列表，每项为 dict：
                              #   {"question": str, "intent": str,
                              #    "rubric": {"excellent": str, "good": str,
                              #               "poor": str}}

    # ---- 面试反馈收集（PRD: collect_feedback，支持多面试官）----
    interview_answers: list    # 候选人回答（兼容旧格式 list[str]）
    interviewer_feedbacks: list  # 多面试官反馈：
                              #   [{"interviewer": str, "scores": [{"question_idx": int,
                              #     "score": int, "comment": str}], "overall": str}]

    # ---- 面评合并（PRD: merge_feedback）----
    merged_feedback: dict     # 合并后的面评分析：
                              #   {"consensus_score": float,
                              #    "disagreements": [{"question_idx": int,
                              #      "scores": [...], "analysis": str}],
                              #    "strengths": [str], "concerns": [str],
                              #    "follow_up_suggestions": [str],
                              #    "summary": str}

    # ---- 面试评估（保留兼容）----
    evaluation: dict           # 面试评估：{"items": [{question, answer, score, comment}],
                              #   "overall_score", "overall_comment"}

    # ---- 录用决策（PRD: decision_gate）----
    recommendation: str        # 录用建议：强烈推荐 / 建议录用 / 谨慎考虑 / 不建议录用
    recommendation_reason: str # 建议理由和风险点

    # ---- Offer 生成（PRD: offer_pack）----
    offer_package: dict       # Offer 信息包：
                              #   {"offer_talking_points": [str],
                              #    "salary_suggestion": str,
                              #    "onboarding_checklist": [str],
                              #    "start_date_suggestion": str}

    # ---- 最终报告 ----
    final_report: str          # report_generator 输出的完整 Markdown 报告

    # ---- 流程控制 ----
    status: str                # running / paused / completed / error / rejected
    current_step: str          # 当前节点名
    hr_decisions: dict         # HR 各环节决定，结构：
                              # {"screening": "approved"|"rejected",
                              #  "screen_route": "phone_screen"|"onsite"|...,
                              #  "interview": "approved"|"rejected",
                              #  "final": "offer"|"rejected"|"continue"}
    messages: Annotated[list, add_messages]  # 消息历史（追加模式）
```

---

## 4. LangGraph 流水线

### 4.1 流程图（含 4 路分流、多面试官反馈和 Offer 路径）

**关键设计**：在每个 HITL 点引入"审核门节点"（gate node）。LangGraph 的 `interrupt_before` 在门节点之前暂停，HR 更新 `hr_decisions` 后恢复，门节点执行后条件边才触发路由，确保拒绝路径正确生效。

```
[START]
   │
   ▼
[parse_jd]                   解析 JD → parsed_jd
   │
   ▼
[resume_parser]              解析简历 → parsed_resume
   │
   ▼
[screener]                   匹配评分 → score, score_reason, evidence, screen_route
   │
   ▼
⏸️ HITL #1                   HR 审核筛选结果，更新 hr_decisions["screening"]
   │
[screening_gate]             读取 HR 决定（pass-through 门节点）
   │
   ├── reject / need_more_info → [report_generator] → [END]
   │
   └── phone_screen / onsite ↓
[interview_generator]        生成面试题（含 rubrics）→ interview_questions
   │
   ▼
⏸️ HITL #2                   HR 审核面试题 + 录入候选人回答
   │                         + 更新 hr_decisions["interview"]
[interview_gate]             读取 HR 决定（pass-through 门节点）
   │
   ├── rejected → [report_generator] → [END]
   │
   └── approved ↓
[evaluator]                  逐题面试评估（含 rubrics 对照）→ evaluation
   │
   ▼
⏸️ HITL #3                   等待多位面试官通过 API 提交反馈
   │                         → interviewer_feedbacks
[collect_feedback]           读取反馈（pass-through HITL 门节点）
   │
   ▼
[merge_feedback]             合并多面试官反馈 → merged_feedback
   │
   ▼
[advisor]                    综合所有信息 → recommendation
   │
   ▼
⏸️ HITL #4                   HR 做最终录用决策
   │                         更新 hr_decisions["final"] = offer/rejected/continue
[decision_gate]              读取 HR 决定（pass-through 门节点）
   │
   ├── offer → [offer_pack] → [report_generator] → [END]
   │
   └── rejected / continue → [report_generator] → [END]
```

### 4.2 智能体节点说明

| # | 节点 | 类型 | 功能 | 输入字段 | 输出字段 |
|---|------|------|------|----------|----------|
| 1 | `parse_jd` | LLM | 解析/规范化岗位画像 | `jd_text` | `parsed_jd` |
| 2 | `resume_parser` | LLM | 结构化提取简历信息 | `resume_text` | `parsed_resume` |
| 3 | `screener` | LLM | 与 JD 比对，逐项评估匹配度，输出 4 路分流建议 | `parsed_jd`, `parsed_resume` | `score`, `score_reason`, `evidence`, `screen_route` |
| 4 | `screening_gate` | Gate | Pass-through 门节点，不调用 LLM | `hr_decisions` | `current_step` |
| 5 | `interview_generator` | LLM | 根据分流类型生成面试题（含 rubrics 评分标准） | `parsed_jd`, `parsed_resume`, `screen_route` | `interview_questions` |
| 6 | `interview_gate` | Gate | Pass-through 门节点，不调用 LLM | `hr_decisions` | `current_step` |
| 7 | `evaluator` | LLM | 逐题评估面试回答，结合 rubrics 打分 | `interview_questions`, `interview_answers`, `jd_text` | `evaluation` |
| 8 | `collect_feedback` | HITL Gate | Pass-through HITL 门节点，等待多面试官提交反馈 | `interviewer_feedbacks` | `current_step` |
| 9 | `merge_feedback` | LLM | 合并多面试官反馈，分析共识与分歧 | `interview_questions`, `interviewer_feedbacks`, `evaluation` | `merged_feedback` |
| 10 | `advisor` | LLM | 综合所有信息给出录用建议 | `parsed_jd`, `parsed_resume`, `score`, `evidence`, `merged_feedback` | `recommendation`, `recommendation_reason` |
| 11 | `decision_gate` | Gate | Pass-through 门节点，HR 做最终录用决策 | `hr_decisions` | `current_step` |
| 12 | `offer_pack` | LLM | 生成 Offer 话术、薪资建议、入职清单（仅 offer 路径） | `parsed_jd`, `parsed_resume`, `score`, `recommendation` | `offer_package` |
| 13 | `report_generator` | 模板 | 汇总所有字段生成 Markdown 报告，不调用 LLM | 全部字段 | `final_report`, `status` |

### 4.3 完整边定义（pipeline.py 实现依据）

```
普通边（固定顺序）：
  START                → parse_jd
  parse_jd             → resume_parser
  resume_parser        → screener
  screener             → screening_gate
  interview_generator  → interview_gate
  evaluator            → collect_feedback
  collect_feedback     → merge_feedback
  merge_feedback       → advisor
  advisor              → decision_gate
  offer_pack           → report_generator
  report_generator     → END

条件边（动态路由，挂载在 gate 节点之后）：
  screening_gate  → route_after_screening()  → interview_generator | report_generator
  interview_gate  → route_after_interview()  → evaluator | report_generator
  decision_gate   → route_after_decision()   → offer_pack | report_generator
```

**为什么需要 gate 节点？**

LangGraph 的执行时序：`interrupt_before` 在节点执行之前暂停。HR 通过 `update_state` 写入 `hr_decisions` 后恢复，gate 节点执行（pass-through），然后条件边触发路由。如果把条件边直接挂在 `screener` 上，条件边会在 HITL 暂停之前就执行（此时 `hr_decisions` 为空），导致拒绝路径永远不生效。

### 4.4 条件边路由函数

```python
def route_after_screening(state: RecruitmentState) -> str:
    """
    筛选后 4 路分流（PRD: route_screen）。
    HR 可通过 hr_decisions.screening 覆盖自动决策。
    """
    hr = state.get("hr_decisions", {})

    # HR 明确拒绝
    if hr.get("screening") == "rejected":
        return "report_generator"

    # HR 明确通过时，使用 HR 指定的分流或 AI 建议
    screen_route = hr.get("screen_route", state.get("screen_route", "phone_screen"))

    if screen_route in ("reject", "need_more_info"):
        return "report_generator"
    else:
        # phone_screen 和 onsite 都进入面试流程
        return "interview_generator"

def route_after_interview(state: RecruitmentState) -> str:
    """面试后路由。HR 通过 hr_decisions.interview 控制。"""
    hr = state.get("hr_decisions", {})
    if hr.get("interview") == "rejected":
        return "report_generator"
    return "evaluator"

def route_after_decision(state: RecruitmentState) -> str:
    """
    最终决策后路由（PRD: decision_gate 出口）。
    HR 通过 hr_decisions.final 控制：
    - offer    → offer_pack → report_generator
    - rejected → report_generator
    - continue → report_generator（可扩展为补面循环）
    """
    hr = state.get("hr_decisions", {})
    final = hr.get("final", "continue")

    if final == "offer":
        return "offer_pack"
    else:
        return "report_generator"
```

**对应 add_conditional_edges / add_edge 调用：**

```python
# 普通边
graph.add_edge(START, "parse_jd")
graph.add_edge("parse_jd", "resume_parser")
graph.add_edge("resume_parser", "screener")
graph.add_edge("screener", "screening_gate")
graph.add_edge("interview_generator", "interview_gate")
graph.add_edge("evaluator", "collect_feedback")
graph.add_edge("collect_feedback", "merge_feedback")
graph.add_edge("merge_feedback", "advisor")
graph.add_edge("advisor", "decision_gate")
graph.add_edge("offer_pack", "report_generator")
graph.add_edge("report_generator", END)

# 条件边（挂载在 gate 节点上）
graph.add_conditional_edges("screening_gate", route_after_screening)
graph.add_conditional_edges("interview_gate", route_after_interview)
graph.add_conditional_edges("decision_gate", route_after_decision)
```

**`current_step` 设置规则**：每个智能体节点在开始执行时将 `current_step` 设为自己的节点名。`report_generator` 在完成后将其设为 `"done"`。

### 4.5 HITL 配置

```python
graph = pipeline.compile(
    checkpointer=MemorySaver(),
    interrupt_before=[
        "screening_gate",       # HITL #1：screener 之后，HR 审核筛选结果，决定分流
        "interview_gate",       # HITL #2：interview_generator 之后，HR 审核问题并提交回答
        "collect_feedback",     # HITL #3：evaluator 之后，等待多面试官提交反馈
        "decision_gate",        # HITL #4：advisor 之后，HR 做最终录用决策
    ]
)
```

---

## 5. LLM JSON 调用策略

所有需要 LLM 返回结构化数据的节点，均通过以下三层机制保证 JSON 输出的可靠性：

### 5.1 API 层：JSON Mode

通过 `get_llm(json_mode=True)` 创建的 LLM 实例会在 API 请求中设置 `response_format: {"type": "json_object"}`，从协议层面强制模型只返回合法 JSON，避免模型在 JSON 前后附加解释性文字或 markdown 代码块。

```python
# tools.py
def get_llm(temperature: float = 0.3, json_mode: bool = False) -> ChatOpenAI:
    kwargs = {}
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(model=..., base_url=..., temperature=temperature, **kwargs)

# agents.py — 所有智能体节点共用 JSON Mode LLM
llm = get_llm(temperature=0.3, json_mode=True)
```

### 5.2 调用层：invoke_for_json 重试封装

所有 LLM 节点通过 `invoke_for_json(llm, messages, max_retries=2)` 调用，而非直接 `llm.invoke()`。该函数封装了：

1. 调用 LLM 并用 `parse_json_from_llm()` 解析响应
2. 若解析失败（返回值含 `_error` 字段），将模型的错误输出追加到对话上下文，要求模型修正
3. 最多重试 `max_retries` 次（默认 2 次，即总共最多调用 3 次）
4. 所有失败均记录 `logging.warning` 日志

```python
# 调用示例（agents.py 中的 8 个 LLM 节点均使用此模式）
parsed = invoke_for_json(llm, [
    SystemMessage(content="...Return ONLY valid JSON..."),
    HumanMessage(content="..."),
])
```

### 5.3 解析层：parse_json_from_llm 多级容错

即使启用了 JSON Mode，解析层仍保留 4 级容错作为兜底：

| 阶段 | 处理 |
|------|------|
| 1. 提取候选文本 | 从 ` ```json ``` ` 代码块或花括号提取 |
| 2. 直接解析 | `json.loads()` 快速路径 |
| 3. 清洗重试 | 修复 Python 布尔值、单引号、尾部逗号、控制字符等 |
| 4. 花括号重提取 | 从原始文本重新定位 `{...}` / `[...]` |
| 兜底 | 返回 `{"_raw": text[:500], "_error": "JSON 解析失败"}` |

---

## 6. FastAPI 接口

### 6.1 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/recruitment/start` | 提交 JD + 简历，创建会话并启动流程 |
| `GET` | `/api/recruitment/{session_id}/state` | 查询当前状态 |
| `POST` | `/api/recruitment/{session_id}/resume` | HR 审核后提交决定，恢复流程 |
| `POST` | `/api/recruitment/{session_id}/answers` | 录入候选人面试回答（HITL #2 专用） |
| `GET` | `/api/recruitment/{session_id}/report` | 获取最终完整报告 |

### 6.2 请求/响应 Pydantic 模型

```python
# ---- 请求模型 ----

class StartRequest(BaseModel):
    jd_text: str       # 岗位要求原文，不得为空
    resume_text: str   # 简历原文，不得为空

class ResumeRequest(BaseModel):
    decision: Literal["approved", "rejected"]  # HR 决定，必填
    notes: str = ""    # HR 备注（可选，会附加到消息历史）

class AnswersRequest(BaseModel):
    answers: list[str]  # 候选人逐题回答，顺序与 interview_questions 对应
                        # 长度必须 >= 1，不要求与问题数完全一致

# ---- 响应模型 ----

class StartResponse(BaseModel):
    session_id: str    # UUID v4，由 POST /start 生成并返回，后续所有操作需携带
    status: Literal["running"]  # 启动成功后状态固定为 running

class StateResponse(BaseModel):
    session_id: str
    status: Literal["running", "paused", "completed", "error", "rejected"]
    paused_at: str | None        # 暂停时为节点名，否则为 null
    current_step: str
    score: int | None            # 仅 screener 完成后非 null
    score_reason: str | None
    interview_questions: list[str] | None   # 仅 interview_generator 完成后非 null
    recommendation: str | None   # 仅 advisor 完成后非 null
    hr_action_required: str | None  # 暂停时提示 HR 需要做什么，枚举值见下表

class ReportResponse(BaseModel):
    session_id: str
    final_report: str            # Markdown 格式的完整报告
    recommendation: str
    hr_decisions: dict
```

**`hr_action_required` 枚举值：**

| `paused_at` 值 | `hr_action_required` 内容 |
|----------------|---------------------------|
| `"interview_generator"` | `"请审核筛选结果（评分+理由），决定是否进入面试流程"` |
| `"evaluator"` | `"请审核面试题，并通过 /answers 接口录入候选人回答后继续"` |
| `"report_generator"` | `"请审核录用建议，做出最终录用决策"` |

### 6.3 HITL #2 接口调用顺序

HITL #2 的正确调用顺序（必须严格执行）：

1. `GET /state` → 返回 `paused_at: "evaluator"`，前端展示面试题
2. HR 确认问题并收集候选人回答后，`POST /answers`（写入 `interview_answers`）
3. `POST /resume { decision: "approved" }` → 恢复流程，执行 `evaluator` 节点

如果调用 `POST /resume` 时 `interview_answers` 为空，API 返回 `422 Unprocessable Entity`。

---

## 7. 错误处理

| 场景 | 处理方式 |
|------|----------|
| LLM 返回非法 JSON | `invoke_for_json` 自动重试最多 2 次（追加错误上下文要求模型修正）；3 次均失败后返回含 `_error` 的兜底 dict，下游节点通过 `isinstance` 检查提供默认值，流程不中断 |
| LLM 调用失败 | 节点捕获异常，`status` 设为 `"error"`，异常信息写入 `messages`；前端调用 `GET /state` 得知 `status=error` 后，通过 `POST /resume { decision:"approved" }` 重新触发同一节点（API 层检测到 `status=error` 时允许 resume 而不视为非法流转） |
| HR 拒绝候选人 | 条件边路由到 `report_generator`，生成拒绝报告，`status` 设为 `"rejected"` |
| 会话 session_id 不存在 | 返回 `404 Not Found` |
| HITL #2 未提交回答就 resume | 返回 `422 Unprocessable Entity` |
| 状态流转非法（如已 completed 再次 resume） | 返回 `409 Conflict` |

**API 执行模型（异步）**：LLM 调用可能耗时较长。`POST /start` 和 `POST /resume` 均使用 FastAPI `BackgroundTasks` 异步执行图，立即返回 `{ session_id, status: "running" }`；前端通过轮询 `GET /state`（每 2 秒一次）感知状态变化，直到 `status` 变为 `paused` 或 `completed`。

---

## 8. 不在本次范围内

- 简历文件解析（PDF/Word）
- 多候选人批量处理
- 数据库持久化（使用 MemorySaver，重启丢失）
- 用户认证/权限管理
- 邮件通知功能

---

## 9. 成功标准

1. 输入简历文本 + JD 文本，能完整走完全部 13 个节点（含 gate 节点）
2. 四个 HITL 暂停点正确暂停，HR 审核后能正确恢复
3. 筛选分流：4 路路由（reject / phone_screen / onsite / need_more_info）正确生效
4. 拒绝路径：任意 HITL 节点返回 `rejected` 后，直接跳转到 `report_generator`
5. Offer 路径：`decision_gate` 返回 `offer` 后，经 `offer_pack` 生成 Offer 信息包
6. FastAPI 所有 5 个接口可正常调用，Pydantic 模型校验生效
7. 前端页面能完整交互完成一次招聘流程（含 HITL 审核步骤）
