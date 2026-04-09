"""
=============================================================
第六课：人机协作（Human-in-the-Loop, HITL）
=============================================================

有时候你不希望 AI 完全自动执行所有操作。
你希望在关键操作前让人类【审批】。

典型场景：
- 发送邮件前需要人类确认
- 执行数据库操作前需要人类审核
- 付款前需要人类批准

LangGraph 通过 interrupt_before / interrupt_after 支持这种模式。
图会在指定的节点处【暂停】，等待人类输入后再【恢复】执行。

流程：
    [起点] --> [智能体] --> [暂停：人类审核] --> [工具] --> [智能体] --> [终点]

本课新概念：
- interrupt_before：在指定节点执行【之前】暂停
- graph.get_state()：暂停时查看当前状态（看 AI 想做什么）
- graph.invoke(None, config)：从暂停处恢复执行
- graph.update_state()：恢复前修改状态（用于拒绝操作）

需要：在 .env 文件中设置 OPENAI_API_KEY
=============================================================
"""

# ============================================================
# 导入模块
# ============================================================

import os                            # 读取环境变量
from typing import Annotated         # 类型标注
from typing_extensions import TypedDict  # 定义状态结构
from dotenv import load_dotenv       # 加载 .env

from langgraph.graph import StateGraph, START, END  # 图核心组件
from langgraph.graph.message import add_messages    # 消息追加模式
from langgraph.checkpoint.memory import MemorySaver # 内存检查点（第五课已学）
from langchain_openai import ChatOpenAI              # OpenAI LLM
from langchain_core.messages import HumanMessage, ToolMessage  # 消息类型
from langchain_core.tools import tool                # 工具装饰器

# 加载环境变量
load_dotenv()


# ============================================================
# 第一步：定义工具（模拟有风险的操作）
# ============================================================
#
# 我们定义两个工具：
# 1. send_email：发送邮件（有风险，需要审批）
# 2. search_web：搜索网页（安全的只读操作）

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to someone. This action requires human approval."""
    # 在真实项目中，这里会真正发送邮件
    # 这种有副作用的操作需要人类审批
    return f"Email sent to {to} with subject '{subject}'"


@tool
def search_web(query: str) -> str:
    """Search the web for information. This is a safe, read-only action."""
    # 搜索是只读操作，相对安全
    return f"Search results for '{query}': [Result 1: LangGraph docs, Result 2: Tutorial]"


# 收集工具并创建查找字典
tools = [send_email, search_web]
tool_map = {t.name: t for t in tools}


# ============================================================
# 第二步：构建智能体图（和第四课类似）
# ============================================================

class AgentState(TypedDict):
    """智能体状态：包含消息列表。"""
    messages: Annotated[list, add_messages]


# 从环境变量读取默认模型名称和 API 地址
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_PROVIDER", "gpt-4o-mini")
BASE_URL = os.getenv("BASE_URL") or None  # 为空则使用 OpenAI 官方地址

# 创建带工具的 LLM
llm = ChatOpenAI(model=DEFAULT_MODEL, base_url=BASE_URL, temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent(state: AgentState) -> dict:
    """智能体节点：调用 LLM 思考并决策。"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def run_tools(state: AgentState) -> dict:
    """工具执行节点：执行 LLM 请求的工具。"""
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool_fn = tool_map[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        print(f"  [工具] 已执行: {tool_call['name']} → {result}")
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],
        ))
    return {"messages": results}


def should_use_tools(state: AgentState) -> str:
    """路由函数：判断是否需要执行工具。"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# 构建图（和第四课一样）
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", run_tools)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_use_tools)
graph_builder.add_edge("tools", "agent")

# ============================================================
# ★ 关键区别：interrupt_before=["tools"]
# ============================================================
# 这告诉 LangGraph：
#   "在执行 tools 节点【之前】暂停图的运行，
#    让人类有机会审核 AI 想要做的操作。"
#
# 也可以用 interrupt_after=["tools"]，
# 表示在执行之后暂停（先执行，再让人类看结果）。

memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,                # 需要检查点来保存暂停时的状态
    interrupt_before=["tools"],          # ★ 在 tools 节点之前暂停！
)


# ============================================================
# 第三步：运行并演示人类审核流程
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 第六课：人机协作（Human-in-the-Loop）")
    print("=" * 60)
    print()

    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-your"):
        print("错误：请在 .env 文件中设置 OPENAI_API_KEY")
        exit(1)

    # 创建一个对话线程
    config = {"configurable": {"thread_id": "hitl_demo_1"}}

    # ========================================================
    # 演示 1：批准操作
    # ========================================================
    print("--- 演示 1：请求一个需要审批的操作 ---")
    print()
    print('  用户: "发一封邮件给 bob@example.com，说声你好"')
    print()

    # 调用 invoke()，图会执行 agent 节点，然后在 tools 节点之前【暂停】
    result = graph.invoke(
        {"messages": [HumanMessage(
            content="Send an email to bob@example.com with subject 'Hello' and body 'Hi Bob, how are you?'"
        )]},
        config=config,
    )

    # ========================================================
    # 图现在处于【暂停】状态。
    # 让我们查看 AI 想做什么，然后决定是否批准。
    # ========================================================
    print("  [已暂停] 智能体想要使用工具。让我们审核一下：")
    print()

    # 通过 get_state() 查看当前状态
    state = graph.get_state(config)
    last_msg = state.values["messages"][-1]

    # 显示 AI 想要调用的工具和参数
    for tc in last_msg.tool_calls:
        print(f"  工具: {tc['name']}")
        print(f"  参数: {tc['args']}")
        print()

    # 在真实项目中，你会把这些信息展示给用户，让用户选择"批准/拒绝"
    # 这里我们模拟自动批准
    print("  [人类] 审核中... 已批准！")
    print()

    # ========================================================
    # 恢复图的执行（从暂停处继续）
    # ========================================================
    print("--- 批准后恢复执行 ---")
    print()

    # 传入 None 表示"没有新的用户输入，继续从暂停处执行"
    result = graph.invoke(None, config=config)

    print()
    print(f"  AI: {result['messages'][-1].content}")
    print()

    # ========================================================
    # 演示 2：拒绝操作
    # ========================================================
    print("=" * 60)
    print("--- 演示 2：拒绝一个可疑操作 ---")
    print()

    # 使用新的线程
    config2 = {"configurable": {"thread_id": "hitl_demo_2"}}

    print('  用户: "发邮件给 spam@bad.com，标题是垃圾广告"')
    print()

    result = graph.invoke(
        {"messages": [HumanMessage(
            content="Send an email to spam@bad.com with subject 'BUY NOW' and body 'Click this link!'"
        )]},
        config=config2,
    )

    print("  [已暂停] 智能体想要发送一封可疑的邮件！")
    print()

    # 查看 AI 想做什么
    state = graph.get_state(config2)
    last_msg = state.values["messages"][-1]

    for tc in last_msg.tool_calls:
        print(f"  工具: {tc['name']}")
        print(f"  参数: {tc['args']}")
    print()

    print("  [人类] 已拒绝！这看起来像垃圾邮件。")
    print()

    # ★ 拒绝的方式：通过 update_state 向状态中注入"拒绝"消息
    # 我们为每个工具调用创建一个 ToolMessage，内容是"已被拒绝"
    # 这样 LLM 恢复执行时会看到拒绝的结果，并做出相应的回应
    rejection_messages = []
    for tc in last_msg.tool_calls:
        rejection_messages.append(ToolMessage(
            content="Action rejected by human reviewer. Do not proceed with this action.",
            tool_call_id=tc["id"],  # 必须关联到对应的工具调用请求
        ))

    # 用 update_state 把拒绝消息注入到状态中
    graph.update_state(config2, {"messages": rejection_messages})

    # 恢复执行——LLM 会看到拒绝消息，并做出适当的回应
    result = graph.invoke(None, config=config2)

    print(f"  AI: {result['messages'][-1].content}")

    # ========================================================
    # 核心要点总结
    # ========================================================
    print()
    print("=" * 60)
    print("核心要点：")
    print("=" * 60)
    print("""
    1. interrupt_before=["节点名"]
       - 在指定节点执行【之前】暂停图的运行
       - 也有 interrupt_after（在执行【之后】暂停）
       - 需要配合 checkpointer 使用（要保存暂停时的状态）

    2. graph.get_state(config)
       - 在暂停时查看当前状态
       - 可以看到 AI 想要调用的工具和参数
       - 供人类审核决策

    3. graph.invoke(None, config)
       - 从暂停处恢复执行
       - 传入 None 表示"继续之前的流程"
       - 工具节点会在恢复后执行

    4. graph.update_state(config, updates)
       - 在恢复之前修改状态
       - 用于拒绝操作：注入"拒绝"消息，LLM 会据此调整行为
       - 也可以用来纠正 AI 的错误

    5. 适用场景
       - 审批敏感操作（邮件、付款、数据库写入）
       - 发布前审核 AI 生成的内容
       - 在执行前纠正 AI 的错误
       - 合规/审计相关的工作流
    """)

    # ========================================================
    # 进阶部分：HITL 的高级模式
    # ========================================================
    print()
    print("=" * 60)
    print("进阶部分：HITL 的高级模式")
    print("=" * 60)

    # ========================================================
    # 进阶 1：interrupt_after —— 执行后暂停
    # ========================================================
    print("""
    ★ 进阶 1：interrupt_after vs interrupt_before

    interrupt_before=["tools"]
    ─────────────────────────────────────────
    智能体 → ⏸️暂停 → [审核] → 工具执行 → 智能体
    用途：执行前审批（"你确定要发这封邮件吗？"）
    ─────────────────────────────────────────

    interrupt_after=["tools"]
    ─────────────────────────────────────────
    智能体 → 工具执行 → ⏸️暂停 → [审核结果] → 智能体
    用途：执行后审核（"工具执行结果如何？需要修改吗？"）
    ─────────────────────────────────────────

    可以同时使用：
    ─────────────────────────────────────────
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["dangerous_tool"],  # 危险操作前暂停
        interrupt_after=["analysis"],         # 分析后让人审核
    )
    ─────────────────────────────────────────

    选择建议：
    • 不可逆操作（发邮件、删数据）→ interrupt_before
    • 可逆操作（搜索、分析）→ interrupt_after
    • 高风险场景 → 两者都用（双重保险）
    """)

    # ========================================================
    # 进阶 2：修改 AI 的决策（纠错）
    # ========================================================
    print("=" * 60)
    print("★ 进阶 2：修改 AI 的决策 —— 不只是批准/拒绝")
    print("=" * 60)
    print("""
    人类审核不仅限于"批准"或"拒绝"。
    你还可以修改 AI 的工具调用参数！

    场景：AI 想发邮件给 wrong@email.com
    人类：收件人不对，应该是 correct@email.com

    ─────────────────────────────────────────
    # 查看 AI 想做什么
    state = graph.get_state(config)
    ai_msg = state.values["messages"][-1]

    # 修改工具调用参数
    ai_msg.tool_calls[0]["args"]["to"] = "correct@email.com"

    # 用修改后的消息替换原始消息
    graph.update_state(
        config,
        {"messages": [ai_msg]},  # 替换（因为 ID 相同）
    )

    # 恢复执行——工具会用修正后的参数
    result = graph.invoke(None, config)
    ─────────────────────────────────────────

    这种模式非常强大：
    • 人类可以微调 AI 的决策而不是完全否决
    • 保留了 AI 的工作流，只修改需要纠正的部分
    • 相当于"AI 打草稿，人类修改后执行"
    """)

    # ========================================================
    # 进阶 3：多步审批工作流
    # ========================================================
    print("=" * 60)
    print("★ 进阶 3：多步审批工作流")
    print("=" * 60)
    print("""
    复杂业务可能需要多个审批点。
    LangGraph 支持在不同节点设置不同的暂停点。

    示例：贷款审批流程
    ─────────────────────────────────────────
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=[
            "credit_check",      # 第一关：信用审核前暂停
            "risk_assessment",   # 第二关：风险评估前暂停
            "final_approval",    # 第三关：最终批准前暂停
        ],
    )

    # 流程：
    # [申请] → ⏸️[审核信用] → [信用检查] → ⏸️[审核风险]
    # → [风险评估] → ⏸️[最终批准] → [放款] → [完成]
    ─────────────────────────────────────────

    每个暂停点可以由不同的人审核：
    • 信用审核 → 风控专员
    • 风险评估 → 风控经理
    • 最终批准 → 信贷总监

    实际项目中的实现方式（配合 Web API）：
    ─────────────────────────────────────────
    # 1. 前端发起请求
    POST /api/loan/start → 流程启动，运行到第一个暂停点

    # 2. 前端轮询状态
    GET /api/loan/{id}/state → 返回当前暂停在哪个节点

    # 3. 审核员做决定后恢复
    POST /api/loan/{id}/resume → 恢复执行到下一个暂停点

    # 4. 重复 2-3 直到流程完成
    ─────────────────────────────────────────
    """)

    # ========================================================
    # 进阶 4：动态中断 —— 运行时决定是否暂停
    # ========================================================
    print("=" * 60)
    print("★ 进阶 4：动态中断 —— 根据条件决定是否暂停")
    print("=" * 60)
    print("""
    有时你不想每次都暂停，只在满足特定条件时暂停。

    方法：使用 NodeInterrupt 异常
    ─────────────────────────────────────────
    from langgraph.errors import NodeInterrupt

    def smart_tool_executor(state):
        tool_call = state["messages"][-1].tool_calls[0]

        # 只对"危险"操作暂停
        if tool_call["name"] in ["send_email", "delete_data"]:
            raise NodeInterrupt(
                f"需要审批: {tool_call['name']}({tool_call['args']})"
            )

        # 安全操作直接执行
        result = tool_map[tool_call["name"]].invoke(tool_call["args"])
        return {"messages": [ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],
        )]}
    ─────────────────────────────────────────

    NodeInterrupt vs interrupt_before：
    • interrupt_before：编译时确定，总是暂停
    • NodeInterrupt：运行时决定，按条件暂停
    • NodeInterrupt 更灵活，适合"有些操作需要审批，有些不需要"
    """)

    # ========================================================
    # 进阶总结
    # ========================================================
    print("=" * 60)
    print("进阶要点总结：")
    print("=" * 60)
    print("""
    1. interrupt_after
       - 执行后暂停，让人类审核结果
       - 适合可逆操作（搜索、分析）
       - 可与 interrupt_before 组合使用

    2. 修改 AI 的决策
       - 不只是批准/拒绝，还可以修改参数
       - update_state 替换消息（利用 ID 匹配）
       - "AI 打草稿，人类修改后执行"模式

    3. 多步审批工作流
       - interrupt_before 可以包含多个节点
       - 不同审批点可由不同角色负责
       - 配合 Web API 实现完整的审批系统

    4. 动态中断 (NodeInterrupt)
       - 运行时根据条件决定是否暂停
       - 比 interrupt_before 更灵活
       - 适合"只对危险操作审批"的场景
    """)
