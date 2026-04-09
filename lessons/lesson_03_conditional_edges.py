"""
=============================================================
第三课：条件边与分支逻辑（不需要 API Key，可以直接运行！）
=============================================================

在第一课和第二课中，图的执行路径是固定的：
    START → A → B → END

但真实的 AI 工作流需要做【决策】！比如：
    - 如果用户问数学题 → 用计算器工具
    - 如果用户说再见 → 结束对话
    - 如果 LLM 不确定 → 要求用户提供更多信息

这就是"条件边"（Conditional Edges）的作用。

本课新概念：
- add_conditional_edges：根据条件动态选择下一个节点的边
- 路由函数（Router Function）：一个返回"下一个节点名称"的函数

示例：一个客服系统，根据用户问题的类型分配到不同的"部门"

    [起点] --> [分类] ---> [技术支持] --> [终点]
                 |                          ^
                 +-------> [账单支持] ------+
                 |                          ^
                 +-------> [通用支持] ------+
=============================================================
"""

# ============================================================
# 导入模块
# ============================================================

# TypedDict：定义状态结构（前两课已学过）
from typing import TypedDict

# StateGraph, START, END：LangGraph 核心组件（前两课已学过）
from langgraph.graph import StateGraph, START, END


# ============================================================
# 第一步：定义 State（状态）
# ============================================================
# 客服系统需要三个字段：
# - query：用户的问题（输入）
# - category：问题的分类（由分类节点填充）
# - response：客服的回复（由各个支持节点填充）

class SupportState(TypedDict):
    """
    客服状态：在图中流转的数据。

    ┌──────────┬───────────────────────────┐
    │ query    │ 用户提的问题（输入）        │
    │ category │ 问题分类（分类节点填写）     │
    │ response │ 客服回复（支持节点填写）     │
    └──────────┴───────────────────────────┘
    """
    query: str           # 用户的问题
    category: str        # 问题分类：technical / billing / general
    response: str        # 客服回复


# ============================================================
# 第二步：定义节点（Nodes）
# ============================================================

def classify(state: SupportState) -> dict:
    """
    分类节点：根据用户问题的关键词判断问题类型。

    在实际项目中，你通常会用 LLM 来做分类（更智能）。
    这里为了演示，我们用简单的关键词匹配。

    分类规则：
    - 包含 bug/error/crash/broken/install → 技术问题
    - 包含 price/bill/pay/refund/cost → 账单问题
    - 其他 → 通用问题
    """
    # 把问题转为小写，方便匹配关键词
    query = state["query"].lower()

    # 根据关键词判断分类
    if any(word in query for word in ["bug", "error", "crash", "broken", "install"]):
        category = "technical"  # 技术问题
    elif any(word in query for word in ["price", "bill", "pay", "refund", "cost"]):
        category = "billing"    # 账单问题
    else:
        category = "general"    # 通用问题

    # 打印分类结果
    print(f"  [分类节点] 问题: '{state['query']}'")
    print(f"  [分类节点] 分类结果: {category}")

    # 返回分类结果，写入 State
    return {"category": category}


def technical_support(state: SupportState) -> dict:
    """
    技术支持节点：处理技术类问题。
    """
    print(f"  [技术支持] 正在处理技术问题...")
    return {
        "response": f"🔧 技术支持：我们已记录您关于 '{state['query']}' 的问题。"
                    f"技术人员将在 24 小时内联系您。"
    }


def billing_support(state: SupportState) -> dict:
    """
    账单支持节点：处理账单类问题。
    """
    print(f"  [账单支持] 正在处理账单问题...")
    return {
        "response": f"💰 账单支持：我们将审核您关于 '{state['query']}' 的账单问题。"
                    f"请查收邮件获取详细回复。"
    }


def general_support(state: SupportState) -> dict:
    """
    通用支持节点：处理其他类型的问题。
    """
    print(f"  [通用支持] 正在处理一般问题...")
    return {
        "response": f"📋 通用支持：感谢您咨询 '{state['query']}'。"
                    f"请查看我们的常见问题页面：https://example.com/faq"
    }


# ============================================================
# 第三步：定义路由函数（Router Function）
# ============================================================
#
# 什么是路由函数？
# ——它是一个普通的 Python 函数，根据当前状态决定"下一步去哪个节点"。
#
# 规则：
# ——它【必须】返回一个字符串，这个字符串必须是某个节点的名称（或 END）。
# ——LangGraph 会根据返回值来决定图的下一步走向。
#
# 类比：
# ——就像路口的交通指挥员，看到不同的车辆，指引它们去不同的方向。

def route_to_department(state: SupportState) -> str:
    """
    路由函数：根据分类结果，决定将问题转发给哪个支持部门。

    参数：state - 当前状态（包含 category 字段）
    返回：下一个节点的名称（字符串）
    """
    category = state["category"]

    if category == "technical":
        return "technical_support"     # 去技术支持节点
    elif category == "billing":
        return "billing_support"       # 去账单支持节点
    else:
        return "general_support"       # 去通用支持节点


# ============================================================
# 第四步：构建带条件边的图
# ============================================================

# 创建图
graph_builder = StateGraph(SupportState)

# 添加所有节点
graph_builder.add_node("classify", classify)                     # 分类节点
graph_builder.add_node("technical_support", technical_support)   # 技术支持节点
graph_builder.add_node("billing_support", billing_support)       # 账单支持节点
graph_builder.add_node("general_support", general_support)       # 通用支持节点

# 添加普通边：起点 → 分类（所有请求都先经过分类）
graph_builder.add_edge(START, "classify")

# ★ 添加条件边：分类 → ??? （由路由函数动态决定！）
# add_conditional_edges 接收两个参数：
#   1. 源节点名称（"classify"）
#   2. 路由函数（route_to_department）
# LangGraph 会调用路由函数，根据返回值决定走哪条边
graph_builder.add_conditional_edges("classify", route_to_department)

# 添加普通边：所有支持节点 → 终点
graph_builder.add_edge("technical_support", END)   # 技术支持 → 终点
graph_builder.add_edge("billing_support", END)     # 账单支持 → 终点
graph_builder.add_edge("general_support", END)     # 通用支持 → 终点

# 编译图
graph = graph_builder.compile()


# ============================================================
# 第五步：运行！
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 第三课：条件边与分支逻辑")
    print("=" * 60)
    print()

    # 准备测试用的问题列表（覆盖三种分类）
    test_queries = [
        "My app keeps crashing on startup",      # → 技术问题（关键词 crash）
        "I want a refund for last month",         # → 账单问题（关键词 refund）
        "What are your business hours?",          # → 通用问题（没有匹配关键词）
        "There's an error when I try to install", # → 技术问题（关键词 error, install）
        "How much does the pro plan cost?",       # → 账单问题（关键词 cost）
    ]

    # 逐个测试
    for query in test_queries:
        print(f"{'=' * 60}")
        result = graph.invoke({
            "query": query,
            "category": "",     # 初始为空，由分类节点填充
            "response": ""      # 初始为空，由支持节点填充
        })
        print(f"  回复: {result['response']}")
        print()

    # ========================================================
    # 图的结构图示
    # ========================================================
    print("=" * 60)
    print("图的结构：")
    print("=" * 60)
    print("""
                    +---> [技术支持] --------+
                    |                        |
    [起点] --> [分类] ---> [账单支持] ---> [终点]
                    |                        |
                    +---> [通用支持] --------+

    分类 → 支持节点 之间的边是【条件边】。
    它使用 route_to_department() 路由函数来决定走哪条路。
    """)

    # ========================================================
    # 核心要点总结
    # ========================================================
    print("=" * 60)
    print("核心要点：")
    print("=" * 60)
    print("""
    1. add_conditional_edges(源节点, 路由函数)
       - 路由函数接收当前状态，返回下一个节点的名称（字符串）
       - 这就是图做"决策"的方式
       - 和普通的 add_edge 不同，它的目标节点不是固定的

    2. 路由函数（Router Function）
       - 就是一个普通的 Python 函数
       - 必须返回一个有效的节点名称或 END
       - 在实际项目中，通常用 LLM 来做路由决策（更智能）

    3. 什么时候用条件边？
       - 根据用户输入路由到不同的处理节点
       - 决定是否需要使用工具
       - 循环回退逻辑（不满意就重试）
       - 任何工作流中的 if/else 决策
    """)

    # ========================================================
    # 进阶部分：条件边的高级用法
    # ========================================================
    print()
    print("=" * 60)
    print("进阶部分：条件边的高级用法")
    print("=" * 60)

    # ========================================================
    # 进阶 1：path_map —— 路由映射表
    # ========================================================
    print("""
    ★ 进阶 1：path_map —— 让路由函数返回值更灵活

    默认情况下，路由函数返回的字符串必须是节点名称。
    但有时路由函数返回的是"分类标签"而非节点名称。
    path_map 可以做映射转换。

    ─────────────────────────────────────────
    def classify(state):
        # 返回的是分类标签，不是节点名称
        return "tech"  # 而不是 "technical_support"

    graph.add_conditional_edges(
        "classify",
        classify,
        path_map={                          # 映射表
            "tech": "technical_support",    # "tech" → 去 technical_support 节点
            "bill": "billing_support",      # "bill" → 去 billing_support 节点
            "other": "general_support",     # "other" → 去 general_support 节点
        }
    )
    ─────────────────────────────────────────

    path_map 的好处：
    • 路由函数可以返回简短的标签（解耦路由逻辑和节点命名）
    • 多个路由值可以映射到同一个节点
    • 让代码更清晰、维护更方便
    """)

    # ========================================================
    # 进阶 2：用 LLM 做智能路由
    # ========================================================
    print("=" * 60)
    print("★ 进阶 2：用 LLM 做智能路由（替代关键词匹配）")
    print("=" * 60)
    print("""
    本课示例用关键词匹配做分类，但实际项目中通常用 LLM。
    LLM 能理解语义，而不仅仅是匹配关键词。

    方法 1：用结构化输出（推荐）
    ─────────────────────────────────────────
    from pydantic import BaseModel, Field

    class RouteDecision(BaseModel):
        \"\"\"路由决策\"\"\"
        department: str = Field(
            description="选择部门",
            enum=["technical", "billing", "general"]
        )
        reason: str = Field(description="选择原因")

    # LLM 结构化输出：保证返回有效的 JSON
    structured_llm = llm.with_structured_output(RouteDecision)

    def llm_classify(state):
        result = structured_llm.invoke(
            f"将以下问题分类到合适的部门：{state['query']}"
        )
        return {"category": result.department}
    ─────────────────────────────────────────

    方法 2：直接用 LLM 返回标签
    ─────────────────────────────────────────
    def llm_route(state):
        response = llm.invoke(
            f"将问题分类，只返回一个词："
            f"technical/billing/general\\n"
            f"问题：{state['query']}"
        )
        return response.content.strip().lower()
    ─────────────────────────────────────────

    方法 1（结构化输出）更可靠，因为 LLM 被限制只能返回
    预定义的选项，不会出现意外的返回值。
    """)

    # ========================================================
    # 进阶 3：多条件组合 —— 复杂路由逻辑
    # ========================================================
    print("=" * 60)
    print("★ 进阶 3：复杂路由模式")
    print("=" * 60)
    print("""
    实际项目中的路由逻辑可能很复杂。以下是常见模式：

    模式 1：多级路由（串联条件边）
    ─────────────────────────────────────────
    [起点] → [一级分类] → [技术] → [二级分类] → [前端/后端/数据库]
                        → [账单] → [终点]
                        → [通用] → [终点]
    ─────────────────────────────────────────

    模式 2：条件边 + END（提前退出）
    ─────────────────────────────────────────
    def route(state):
        if state["score"] < 30:
            return END              # 分数太低，直接终止
        elif state["score"] < 70:
            return "review"         # 需要进一步审核
        else:
            return "auto_approve"   # 自动通过

    graph.add_conditional_edges("score_check", route)
    ─────────────────────────────────────────

    模式 3：循环条件（迭代改进）
    ─────────────────────────────────────────
    def should_retry(state):
        if state["quality"] >= 0.8:
            return END             # 质量达标，结束
        if state["attempts"] >= 3:
            return END             # 尝试够了，结束
        return "improve"           # 继续改进

    graph.add_conditional_edges("evaluate", should_retry)
    graph.add_edge("improve", "evaluate")  # 循环
    ─────────────────────────────────────────

    这个"循环条件"模式在第四课（智能体循环）中会深入学习。
    """)

    # ========================================================
    # 进阶 4：条件边的调试技巧
    # ========================================================
    print("=" * 60)
    print("★ 进阶 4：条件边的调试技巧")
    print("=" * 60)
    print()

    # 用 stream() 查看实际路由路径
    print("  使用 stream() 追踪路由路径：")
    print()

    test_query = "My computer keeps crashing"
    for step in graph.stream({
        "query": test_query,
        "category": "",
        "response": ""
    }):
        for node_name in step:
            print(f"    → 经过节点: {node_name}")

    print()
    print("  其他调试技巧：")
    print("    1. 在路由函数中添加 print() 打印决策过程")
    print("    2. 用 graph.get_graph().draw_mermaid() 可视化图结构")
    print("    3. 用 stream() 逐步追踪执行路径")
    print("    4. 确保路由函数返回的字符串与节点名称完全匹配")
    print()

    # ========================================================
    # 进阶总结
    # ========================================================
    print("=" * 60)
    print("进阶要点总结：")
    print("=" * 60)
    print("""
    1. path_map 路由映射
       - 解耦路由标签和节点名称
       - 让路由函数更灵活、代码更清晰

    2. LLM 智能路由
       - 用 with_structured_output() 保证输出格式
       - 比关键词匹配更智能、更鲁棒
       - 实际项目的首选方案

    3. 复杂路由模式
       - 多级路由、提前退出、循环条件
       - 条件边 + END = 提前终止
       - 条件边 + 循环 = 迭代改进

    4. 调试技巧
       - stream() 追踪路由路径
       - Mermaid 可视化图结构
       - 路由函数中添加日志
    """)
