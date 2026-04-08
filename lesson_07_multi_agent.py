"""
=============================================================
第七课：多智能体协作（Multi-Agent Collaboration）
=============================================================

这是最高级的模式：多个专业化的智能体协同工作，
每个智能体有自己的专长。

把它想象成一个团队：
- 研究员（Researcher）：擅长搜集信息
- 写手（Writer）：擅长撰写内容
- 审核员（Reviewer）：擅长检查质量

LangGraph 让你把这种团队协作构建为一个图：
- 每个智能体是一个节点
- 一个"主管"（Supervisor）决定谁来干活
- 智能体通过共享状态传递成果

模式：主管架构（Supervisor Architecture）

    [起点] --> [主管] --> [研究员] --+
                ^  |                 |
                |  +---> [写手] -----+
                |  |                 |
                |  +---> [审核员] ---+
                |                    |
                +---- 循环回来 ------+
                |
                +--> [终点]（主管说"完成了"时）

需要：在 .env 文件中设置 OPENAI_API_KEY
=============================================================
"""

# ============================================================
# 导入模块
# ============================================================

import os                                  # 读取环境变量
from typing import Annotated, Literal      # 类型标注工具
from typing_extensions import TypedDict    # 定义状态结构
from dotenv import load_dotenv             # 加载 .env 文件

from langgraph.graph import StateGraph, START, END  # 图核心组件
from langgraph.graph.message import add_messages    # 消息追加模式
from langchain_openai import ChatOpenAI              # OpenAI LLM 封装

# 消息类型：
# HumanMessage - 用户消息
# AIMessage - AI 回复
# SystemMessage - 系统提示词（用于设定智能体的"人设"和"角色"）
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 加载环境变量
load_dotenv()


# ============================================================
# 第一步：定义共享状态（Shared State）
# ============================================================
#
# 多智能体的核心：所有智能体共享同一个 State。
# 每个智能体可以读取其他智能体写入的数据，
# 也可以把自己的产出写进 State。
#
# 这就像一块共享白板：
# - 研究员在白板上写下调研结果
# - 写手看到调研结果后，在白板上写下初稿
# - 审核员看到初稿后，在白板上写下审核意见
# - 主管看全部内容，决定下一步做什么

class TeamState(TypedDict):
    """
    团队共享状态：所有智能体都可以读写这些字段。

    ┌──────────┬────────────────────────────────────┐
    │ messages  │ 对话历史（追加模式）                 │
    │ task      │ 当前任务描述                        │
    │ research  │ 研究员的调研结果                     │
    │ draft     │ 写手的初稿                          │
    │ review    │ 审核员的审核意见                     │
    │ next_agent│ 主管决定的下一个智能体                │
    │ iteration │ 已经进行了多少轮                     │
    └──────────┴────────────────────────────────────┘
    """
    messages: Annotated[list, add_messages]  # 对话历史（追加模式）
    task: str              # 任务描述
    research: str          # 研究员的调研结果
    draft: str             # 写手撰写的初稿
    review: str            # 审核员的审核意见
    next_agent: str        # 下一个应该工作的智能体
    iteration: int         # 迭代轮数（防止无限循环）


# ============================================================
# 第二步：创建 LLM 实例
# ============================================================
# 所有智能体共用一个 LLM，但通过不同的 SystemMessage 来区分角色。
# 在更复杂的项目中，不同智能体也可以使用不同的模型。

# 从环境变量读取默认模型名称和 API 地址
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_PROVIDER", "gpt-4o-mini")
BASE_URL = os.getenv("BASE_URL") or None  # 为空则使用 OpenAI 官方地址

llm = ChatOpenAI(model=DEFAULT_MODEL, base_url=BASE_URL, temperature=0.7)


# ============================================================
# 第三步：定义各个智能体节点
# ============================================================

def supervisor(state: TeamState) -> dict:
    """
    主管节点：决定下一步该由哪个智能体来工作。

    决策逻辑：
    - 如果还没有调研结果 → 派研究员去调研
    - 如果有了调研结果但没有初稿 → 派写手去写
    - 如果有了初稿但没有审核意见 → 派审核员去审核
    - 如果审核说"需要改进"且迭代次数不够 → 让写手修改
    - 否则 → 任务完成

    在真实项目中，你通常会用 LLM 来做这个决策（更智能更灵活）。
    这里为了清晰演示，我们用简单的规则逻辑。
    """
    print(f"\n  [主管] 正在分析当前进度（第 {state.get('iteration', 0)} 轮）...")

    # 获取当前迭代次数
    iteration = state.get("iteration", 0)

    # 根据当前状态决定下一个智能体
    if not state.get("research"):
        # 还没有调研结果 → 让研究员去调研
        next_agent = "researcher"
    elif not state.get("draft"):
        # 有了调研但还没写初稿 → 让写手去写
        next_agent = "writer"
    elif not state.get("review"):
        # 有了初稿但还没审核 → 让审核员去审核
        next_agent = "reviewer"
    elif iteration < 2 and "needs improvement" in state.get("review", "").lower():
        # 审核员说"需要改进"，且迭代次数不够 → 打回给写手修改
        next_agent = "writer"
    else:
        # 所有工作都完成了，或者已经迭代够了 → 结束
        next_agent = "FINISH"

    print(f"  [主管] 决策: {next_agent}")
    return {"next_agent": next_agent, "iteration": iteration + 1}


def researcher(state: TeamState) -> dict:
    """
    研究员节点：根据任务主题搜集关键信息。

    通过 SystemMessage 设定角色：告诉 LLM "你是一个研究专家"。
    """
    print(f"  [研究员] 正在调研: {state['task']}")

    # 用 SystemMessage 设定角色人设
    # 用 HumanMessage 传递具体的调研任务
    response = llm.invoke([
        SystemMessage(content=(
            "You are a research specialist. Your job is to gather key facts "
            "and information about the given topic. Be concise and factual. "
            "Provide 3-5 key points."
        )),
        HumanMessage(content=f"Research this topic: {state['task']}"),
    ])

    research = response.content
    print(f"  [研究员] 调研完成，找到了关键信息")

    # 返回调研结果（写入共享状态）
    return {
        "research": research,
        "messages": [AIMessage(content=f"[研究员] {research}")],
    }


def writer(state: TeamState) -> dict:
    """
    写手节点：根据调研结果撰写内容。

    如果有审核员的反馈，会根据反馈修改文章。
    """
    print(f"  [写手] 正在根据调研结果撰写...")

    # 构建写作提示
    prompt = f"Write a short article about: {state['task']}\n\n"
    prompt += f"Research findings:\n{state['research']}\n\n"

    # 如果有审核反馈，加入到提示中，让写手根据反馈修改
    if state.get("review"):
        prompt += f"Previous review feedback:\n{state['review']}\n"
        prompt += "Please address the feedback in your revision.\n"

    response = llm.invoke([
        SystemMessage(content=(
            "You are a professional writer. Write clear, engaging content "
            "based on the research provided. Keep it under 150 words."
        )),
        HumanMessage(content=prompt),
    ])

    draft = response.content
    print(f"  [写手] 初稿完成")

    return {
        "draft": draft,
        "messages": [AIMessage(content=f"[写手] {draft}")],
    }


def reviewer(state: TeamState) -> dict:
    """
    审核员节点：审核初稿的质量。

    审核标准：清晰度、准确性、吸引力。
    如果质量好，以 "Approved:" 开头。
    如果需要改进，以 "Needs improvement:" 开头。
    """
    print(f"  [审核员] 正在审核初稿...")

    response = llm.invoke([
        SystemMessage(content=(
            "You are a content reviewer. Review the following draft for "
            "clarity, accuracy, and engagement. Provide brief feedback. "
            "If the draft is good, start with 'Approved:'. "
            "If it needs work, start with 'Needs improvement:'."
        )),
        HumanMessage(content=f"Review this draft:\n\n{state['draft']}"),
    ])

    review = response.content
    print(f"  [审核员] 审核完成")

    return {
        "review": review,
        "messages": [AIMessage(content=f"[审核员] {review}")],
    }


# ============================================================
# 第四步：主管的路由函数
# ============================================================

def route_supervisor(state: TeamState) -> str:
    """
    路由函数：根据主管的决策，路由到对应的智能体节点或终点。
    """
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return END        # 任务完成，去终点
    return next_agent     # 去对应的智能体节点


# ============================================================
# 第五步：构建多智能体图
# ============================================================

graph_builder = StateGraph(TeamState)

# 添加所有智能体节点
graph_builder.add_node("supervisor", supervisor)     # 主管节点
graph_builder.add_node("researcher", researcher)     # 研究员节点
graph_builder.add_node("writer", writer)             # 写手节点
graph_builder.add_node("reviewer", reviewer)         # 审核员节点

# 起点 → 主管（所有请求先经过主管）
graph_builder.add_edge(START, "supervisor")

# 主管 → 各智能体 或 终点（条件边，由路由函数决定）
graph_builder.add_conditional_edges("supervisor", route_supervisor)

# 所有智能体完成任务后 → 回到主管（汇报成果，等待下一步指示）
graph_builder.add_edge("researcher", "supervisor")  # 研究员 → 主管
graph_builder.add_edge("writer", "supervisor")      # 写手 → 主管
graph_builder.add_edge("reviewer", "supervisor")    # 审核员 → 主管

# 编译图
graph = graph_builder.compile()


# ============================================================
# 第六步：运行！
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 第七课：多智能体协作")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-your"):
        print("错误：请在 .env 文件中设置 OPENAI_API_KEY")
        exit(1)

    # 给团队分配一个任务
    task = "The benefits and risks of using AI in education"

    print(f"\n  任务: {task}")
    print(f"  团队: 主管、研究员、写手、审核员")
    print()
    print("-" * 60)

    # 运行图
    result = graph.invoke({
        "messages": [HumanMessage(content=f"Please create content about: {task}")],
        "task": task,
        "research": "",       # 初始为空，由研究员填充
        "draft": "",          # 初始为空，由写手填充
        "review": "",         # 初始为空，由审核员填充
        "next_agent": "",     # 初始为空，由主管决定
        "iteration": 0,       # 从第 0 轮开始
    })

    # ========================================================
    # 展示最终结果
    # ========================================================
    print()
    print("=" * 60)
    print("最终结果：")
    print("=" * 60)

    print("\n--- 调研结果 ---")
    print(result["research"][:500])

    print("\n--- 最终稿件 ---")
    print(result["draft"][:500])

    print("\n--- 审核意见 ---")
    print(result["review"][:500])

    print(f"\n--- 总迭代轮数: {result['iteration']} ---")

    # ========================================================
    # 核心要点总结
    # ========================================================
    print()
    print("=" * 60)
    print("核心要点：")
    print("=" * 60)
    print("""
    1. 主管模式（Supervisor Pattern）
       - 一个"主管"节点决定下一个工作的智能体
       - 每个智能体完成任务后都向主管汇报
       - 主管可以路由到任意智能体或终点

    2. 共享状态（Shared State）
       - 所有智能体读写同一个 State
       - 研究员写入 "research"，写手读取它
       - State 就是智能体之间协作的"共享白板"

    3. 迭代改进
       - 主管可以把工作打回去让智能体修改
       - 审核员 → 主管 → 写手 → 主管 → 审核员
       - 形成"反馈循环"，不断提升质量

    4. 什么时候用多智能体？
       - 复杂任务需要不同领域的专业知识
       - 任务有明确的阶段（调研 → 写作 → 审核）
       - 你想让不同的角色各司其职（关注点分离）

    5. 其他多智能体模式
       - 层级式：主管管理子主管
       - 对等式：智能体之间直接对话
       - 流水线式：固定顺序 A → B → C
    """)
