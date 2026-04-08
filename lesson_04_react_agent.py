"""
=============================================================
第四课：工具调用 - 构建一个 ReAct Agent（智能体）
=============================================================

这是 LangGraph 中【最重要】的模式！

什么是 ReAct Agent？
——一个能"思考 + 行动"的 AI 智能体：
1. 思考（Reason）：LLM 分析问题，决定该做什么
2. 行动（Act）：调用工具（函数）获取信息
3. 再思考：LLM 查看工具返回的结果
4. 重复以上步骤，直到找到答案

图的结构（注意有个【循环】！）：

    [起点] --> [agent 智能体] -----> [tools 工具] --+
                   |        ^                      |
                   |        +------ 循环回来 ------+
                   |
                   +---> (不需要工具) --> [终点]

这就是"智能体循环"（Agentic Loop）：
- agent 节点调用 LLM
- LLM 判断："我需要用工具" 或 "我已经有答案了"
- 如果需要工具 → 执行工具 → 把结果反馈给 agent
- 如果有答案了 → 直接到终点

本课新概念：
- @tool 装饰器：定义 LLM 可以使用的工具
- bind_tools：把工具绑定给 LLM（让 LLM 知道有哪些工具可用）
- ToolMessage：工具执行结果的消息类型
- 智能体循环模式（Agentic Loop）

需要：在 .env 文件中设置 OPENAI_API_KEY
=============================================================
"""

# ============================================================
# 导入模块
# ============================================================

import os     # 读取环境变量
import json   # JSON 处理（调试用）

# Annotated：标注类型，用于追加模式（第二课已学过）
from typing import Annotated
# TypedDict：定义状态结构（第一课已学过）
from typing_extensions import TypedDict
# load_dotenv：加载 .env 文件中的环境变量
from dotenv import load_dotenv

# LangGraph 核心组件
from langgraph.graph import StateGraph, START, END
# add_messages：消息追加模式（第二课已学过）
from langgraph.graph.message import add_messages
# ChatOpenAI：OpenAI 聊天模型封装（第二课已学过）
from langchain_openai import ChatOpenAI

# 消息类型：
# HumanMessage - 用户消息
# AIMessage - AI 回复
# ToolMessage - 工具执行结果（本课新学！）
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# @tool 装饰器：把普通 Python 函数变成 LLM 可调用的"工具"
from langchain_core.tools import tool

# 加载环境变量
load_dotenv()


# ============================================================
# 第一步：定义工具（Tools）
# ============================================================
#
# 什么是工具？
# ——工具就是你写的普通 Python 函数，但通过 @tool 装饰器，
#   LLM 就能"知道"这个函数的存在，并在需要时"请求调用"它。
#
# 重要理解：
# ——LLM 并不会直接运行函数！
#   它只是生成一个"工具调用请求"，比如：
#   "请帮我运行 add 函数，参数是 a=3, b=5"
#   然后由我们的代码真正执行函数，并把结果发回给 LLM。
#
# @tool 装饰器的作用：
# 1. 把函数的名字、参数、文档字符串（docstring）打包成 LLM 能理解的格式
# 2. LLM 会根据 docstring 来判断"什么时候该用这个工具"
#    所以 docstring 写得清晰很重要！

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together. Use this for addition."""
    # 加法：把两个数加在一起
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers. Use this for multiplication."""
    # 乘法：把两个数相乘
    return a * b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this to check weather."""
    # 获取城市天气（这里用假数据模拟，实际项目中会调用真实的天气 API）
    weather_data = {
        "beijing": "晴天, 25°C",
        "shanghai": "多云, 22°C",
        "tokyo": "下雨, 18°C",
        "new york": "刮风, 15°C",
        "london": "大雾, 12°C",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather in {city}: Partly cloudy, 20°C (default)"


# 把所有工具收集到一个列表中
tools = [add, multiply, get_weather]

# 创建一个查找字典：工具名称 → 工具函数
# 当 LLM 请求调用某个工具时，我们需要通过名字找到对应的函数
# 例如：tool_map["add"] 就能找到 add 这个工具
tool_map = {t.name: t for t in tools}


# ============================================================
# 第二步：创建带工具的 LLM
# ============================================================
#
# bind_tools() 是关键方法！
# 它告诉 LLM："你有这些工具可以用。"
# LLM 会看到每个工具的名字、参数列表和文档字符串，
# 然后在合适的时候选择调用它们。
#
# temperature=0 表示确定性输出（不随机），
# 这样工具调用的结果更稳定可靠。

# 从环境变量读取默认模型名称和 API 地址
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_PROVIDER", "gpt-4o-mini")
BASE_URL = os.getenv("BASE_URL") or None  # 为空则使用 OpenAI 官方地址

llm = ChatOpenAI(model=DEFAULT_MODEL, base_url=BASE_URL, temperature=0)

# ★ 关键步骤：把工具绑定给 LLM
# 绑定后，LLM 在生成回复时就可以选择"调用工具"而不是"直接回答"
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 第三步：定义状态
# ============================================================
class AgentState(TypedDict):
    """
    智能体状态：包含对话消息列表。
    所有消息（用户消息、AI回复、工具结果）都存在这个列表中。
    """
    messages: Annotated[list, add_messages]  # 消息列表（追加模式）


# ============================================================
# 第四步：定义节点
# ============================================================

def agent(state: AgentState) -> dict:
    """
    智能体节点：调用 LLM 进行思考。

    LLM 会分析对话历史，然后做出两种决策之一：
    a) 返回普通文本回复 → 表示已经有了最终答案
    b) 返回一个或多个"工具调用请求" → 表示需要先用工具获取信息

    我们可以通过 response.tool_calls 来判断 LLM 的决策。
    """
    print("  [智能体] 正在思考...")

    # 把所有消息发给 LLM，让它决定下一步该做什么
    response = llm_with_tools.invoke(state["messages"])

    # 检查 LLM 是否想要调用工具
    if response.tool_calls:
        # LLM 想要使用工具，打印它想调用的工具信息
        for tc in response.tool_calls:
            print(f"  [智能体] 想要使用工具: {tc['name']}({tc['args']})")
    else:
        # LLM 直接给出了最终答案
        print(f"  [智能体] 最终答案已就绪")

    # 返回 LLM 的回复（追加到消息列表）
    return {"messages": [response]}


def run_tools(state: AgentState) -> dict:
    """
    工具执行节点：运行 LLM 请求的工具。

    工作流程：
    1. 从最后一条消息中读取 LLM 的工具调用请求
    2. 逐个执行请求的工具
    3. 把每个工具的执行结果包装成 ToolMessage
    4. 返回结果（追加到消息列表，这样 LLM 下次就能看到结果了）
    """
    # 获取最后一条消息（应该包含 tool_calls）
    last_message = state["messages"][-1]

    results = []
    for tool_call in last_message.tool_calls:
        # 从工具调用请求中提取工具名称和参数
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"  [工具] 正在执行 {tool_name}({tool_args})")

        # 通过名称查找对应的工具函数，然后执行
        tool_fn = tool_map[tool_name]
        result = tool_fn.invoke(tool_args)

        print(f"  [工具] 执行结果: {result}")

        # 把结果包装成 ToolMessage
        # tool_call_id 必须和请求的 id 匹配，
        # 这样 LLM 才能知道"这是哪个工具调用的结果"
        results.append(ToolMessage(
            content=str(result),               # 工具的执行结果
            tool_call_id=tool_call["id"],       # 关联到对应的请求 ID
        ))

    # 返回所有工具结果（追加到消息列表）
    return {"messages": results}


# ============================================================
# 第五步：定义路由函数（决定是用工具还是结束）
# ============================================================

def should_use_tools(state: AgentState) -> str:
    """
    路由函数：检查 LLM 的最新回复是否包含工具调用请求。

    返回值：
      "tools" → LLM 想用工具（去工具执行节点）
      END     → LLM 已有最终答案（去终点，结束）
    """
    # 获取最后一条消息
    last_message = state["messages"][-1]

    # 如果 LLM 的回复中有 tool_calls，说明它想调用工具
    if last_message.tool_calls:
        return "tools"  # 去工具执行节点

    # 否则，LLM 已经给出了最终答案
    return END  # 去终点，结束


# ============================================================
# 第六步：构建图（智能体循环！）
# ============================================================

graph_builder = StateGraph(AgentState)

# 添加节点
graph_builder.add_node("agent", agent)       # 智能体节点（调用 LLM）
graph_builder.add_node("tools", run_tools)   # 工具执行节点（运行工具）

# 起点 → 智能体（总是从 LLM 开始）
graph_builder.add_edge(START, "agent")

# 智能体 → 工具 或 终点（条件边！由路由函数决定）
graph_builder.add_conditional_edges("agent", should_use_tools)

# ★ 工具 → 智能体（循环回来！LLM 会看到工具的执行结果）
# 这条边形成了一个【循环】，是智能体模式的核心！
# 工具执行完后，结果会反馈给 LLM��LLM 再决定下一步
graph_builder.add_edge("tools", "agent")

# 编译图
graph = graph_builder.compile()


# ============================================================
# 第七步：运行！
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 第四课：ReAct 智能体（工具调用）")
    print("=" * 60)
    print()

    # 检查 API Key
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-your"):
        print("错误：请在 .env 文件中设置 OPENAI_API_KEY")
        exit(1)

    # ========================================================
    # 测试 1：数学问题（需要调用工具）
    # ========================================================
    print("--- 测试 1：数学问题 ---")
    print()

    result = graph.invoke({
        "messages": [HumanMessage(content="What is 15 + 27?")]
    })
    print(f"\n  最终答案: {result['messages'][-1].content}")
    print()

    # ========================================================
    # 测试 2：天气查询（需要调用工具）
    # ========================================================
    print("--- 测试 2：天气查询 ---")
    print()

    result = graph.invoke({
        "messages": [HumanMessage(content="What's the weather in Beijing?")]
    })
    print(f"\n  最终答案: {result['messages'][-1].content}")
    print()

    # ========================================================
    # 测试 3：多步推理（需要多次调用工具）
    # ========================================================
    print("--- 测试 3：多步推理 ---")
    print()

    result = graph.invoke({
        "messages": [HumanMessage(
            content="What is 3 + 5, and then multiply that result by 4?"
        )]
    })
    print(f"\n  最终答案: {result['messages'][-1].content}")
    print()

    # ========================================================
    # 测试 4：不需要工具的简单问题
    # ========================================================
    print("--- 测试 4：简单问题（不需要工具）---")
    print()

    result = graph.invoke({
        "messages": [HumanMessage(content="Hello, how are you?")]
    })
    print(f"\n  最终答案: {result['messages'][-1].content}")

    # ========================================================
    # 智能体循环示意图
    # ========================================================
    print()
    print("=" * 60)
    print("智能体循环（Agentic Loop）：")
    print("=" * 60)
    print("""
    [起点] --> [智能体 (LLM)] -----> [工具 (执行)] --+
                   |          ^                      |
                   |          +------ 循环回来 ------+
                   |
                   +---> (不需要工具) --> [终点]

    这个循环可以重复多次，例如：

    第 1 轮: "3+5等于几？" → LLM 说 "请调用 add(3,5)"
             → 工具返回 8 → 反馈给 LLM
    第 2 轮: LLM 说 "请调用 multiply(8,4)"
             → 工具返回 32 → 反馈给 LLM
    第 3 轮: LLM 说 "答案是 32" → 到达终点
    """)

    print("=" * 60)
    print("核心要点：")
    print("=" * 60)
    print("""
    1. @tool 装饰器
       - 把普通 Python 函数变成 LLM 可以调用的工具
       - 函数的 docstring 告诉 LLM "什么时候该用这个工具"
       - 所以 docstring 一定要写得清晰！

    2. llm.bind_tools(tools)
       - 给 LLM 一个"工具箱"
       - LLM 可以看到每个工具的名称、参数和说明
       - 然后在需要时选择合适的工具

    3. 智能体循环（Agentic Loop）
       - 智能体（LLM）→ 思考决定做什么
       - 工具（执行）→ 运行工具获取结果
       - 循环回来 → LLM 看到结果后继续思考
       - 重复直到完成

    4. 这就是所有 AI 智能体的基础！
       - ChatGPT 的插件？同样的模式
       - Claude 的工具调用？同样的模式
       - 任何能搜索/计算/操作的 AI 助手都是这个模式
    """)
