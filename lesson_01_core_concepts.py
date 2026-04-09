"""
=============================================================
第一课：LangGraph 核心概念（不需要 API Key，可以直接运行！）
=============================================================

LangGraph 是一个用"图"（流程图）来构建 AI 工作流的框架。

把它想象成一个工厂的流水线：
- 数据从起点出发
- 经过一个个"工位"（节点）进行加工处理
- 最终到达终点，输出结果

本课教你 LangGraph 最核心的 4 个概念：

1. State（状态）    - 在图中流转的共享数据容器，类似一个"共享记事本"
2. Node（节点）     - 一个处理数据的函数，类似流水线上的"工位/工人"
3. Edge（边）       - 节点之间的连接线，定义数据的流向（谁先谁后）
4. StateGraph（图） - 把节点和边组合在一起的整体结构

本课示例的流程图：
    [起点 START] --> [greet 打招呼] --> [add_emoji 加表情] --> [终点 END]
=============================================================
"""

# ============================================================
# 第一步：导入需要的模块
# ============================================================

# TypedDict 是 Python 内置的一个类型工具
# 它让我们可以定义一个"有固定字段和类型的字典"
# 你可以把它理解为一个"表单模板"——规定了有哪些字段、每个字段是什么类型
# 例如：{"name": str, "age": int} 表示一个有 name（字符串）和 age（整数）的表单
from typing import TypedDict

# StateGraph：LangGraph 的核心类，用来创建和管理图
# START：特殊标记，表示"图的起点"（数据从这里开始流动）
# END：特殊标记，表示"图的终点"（数据到这里停止流动）
from langgraph.graph import StateGraph, START, END


# ============================================================
# 第二步：定义 State（状态）
# ============================================================
#
# 什么是 State？
# ——State 就像一个"共享记事本"。图中每个节点都可以读取和修改它。
#
# 为什么需要 State？
# ——因为节点之间需要传递数据。比如第一个节点生成了一段问候语，
#   第二个节点需要读取这段问候语来加上表情。
#   State 就是它们之间的"数据桥梁"。
#
# 怎么定义 State？
# ——继承 TypedDict，声明字段名和类型，就像设计一张表格。
#   每个字段就是表格中的一列。
#
# 下面的 GreetingState 有两个字段：
#   - name：要问候的人的名字（字符串类型）
#   - greeting：问候语（字符串类型，初始为空，后续由节点填充）

class GreetingState(TypedDict):
    """
    问候状态：在图中流转的数据结构。

    就像一张有两个空格的表格：
    ┌──────────┬──────────────────────────────┐
    │ name     │ （要问候的人的名字）           │
    │ greeting │ （生成的问候语，由节点逐步填写）│
    └──────────┴──────────────────────────────┘
    """
    name: str       # 用户的名字（输入数据）
    greeting: str   # 问候语（由节点生成和修改）


# ============================================================
# 第三步：定义 Node（节点）
# ============================================================
#
# 什么是 Node？
# ——Node 就是一个普通的 Python 函数！没有什么特殊的。
#   它做两件事：
#   1. 接收当前的 State 作为输入参数（读取共享记事本）
#   2. 返回一个字典，包含它想要更新的字段（修改共享记事本）
#
# 重要规则：
# ——你只需要返回你想"修改"的字段，不需要返回整个 State！
#   比如你只想修改 greeting，就只返回 {"greeting": "新值"}，
#   name 字段会自动保持不变。
#
# 类比：
# ——就像流水线上的工人，每个工人只负责自己那一步的加工。
#   第一个工人写上问候语，第二个工人给问候语加上表情，各司其职。

def greet(state: GreetingState) -> dict:
    """
    节点 1：生成问候语。

    功能：读取名字 → 生成问候语 → 写回状态
    读取：state["name"]（从状态中读取名字）
    写入：state["greeting"]（把生成的问候语写回状态）
    """
    # 从 State 中读取名字
    # state 就是那个"共享记事本"，用字典的方式读取字段
    name = state["name"]

    # 用名字生成一条问候语
    greeting = f"你好，{name}！欢迎学习 LangGraph！"

    # 打印日志，方便我们在终端看到这个节点的执行过程
    print(f"  [greet 节点] 为 {name} 生成了问候语")

    # 返回要更新的字段
    # 注意：只返回 {"greeting": ...}，不需要返回 name
    # LangGraph 会自动把这个返回值合并（merge）到 State 中
    return {"greeting": greeting}


def add_emoji(state: GreetingState) -> dict:
    """
    节点 2：给问候语加上 emoji 表情。

    功能：读取当前问候语 → 加上表情 → 写回状态
    读取：state["greeting"]（从状态中读取当前的问候语）
    写入：state["greeting"]（把加了表情的新问候语写回状态，覆盖旧值）
    """
    # 从 State 中读取上一个节点生成的问候语
    old_greeting = state["greeting"]

    # 在问候语前后加上 emoji 表情，生成新的问候语
    new_greeting = f"🎉 {old_greeting} 🎉"

    # 打印日志
    print(f"  [add_emoji 节点] 给问候语加上了表情")

    # 返回更新后的问候语（会覆盖 State 中的 greeting 字段）
    return {"greeting": new_greeting}


# ============================================================
# 第四步：构建 Graph（图）
# ============================================================
#
# 现在我们把所有东西连接在一起，就像画一张流程图。
# 构建图分 4 个小步骤：
#   4a. 创建空图
#   4b. 添加节点
#   4c. 添加边（连接节点）
#   4d. 编译图

# 4a. 创建一个新的图，告诉它使用什么样的 State 结构
# ——StateGraph(GreetingState) 表示：这个图中流转的数据格式是 GreetingState
graph_builder = StateGraph(GreetingState)

# 4b. 添加节点（给每个节点起一个名字，并绑定对应的函数）
# ——第一个参数是节点的名字（字符串），第二个参数是对应的函数
# ——这个名字后面会用来连接边
graph_builder.add_node("greet", greet)           # 注册 greet 函数为 "greet" 节点
graph_builder.add_node("add_emoji", add_emoji)   # 注册 add_emoji 函数为 "add_emoji" 节点

# 4c. 添加边（定义数据的流向，即节点之间的连接关系）
# ——add_edge(源节点, 目标节点) 表示"从源节点流向目标节点"
# ——START 和 END 是 LangGraph 提供的内置特殊标记
# ——执行顺序就是：START → greet → add_emoji → END
graph_builder.add_edge(START, "greet")           # 起点 → greet（图从 greet 开始执行）
graph_builder.add_edge("greet", "add_emoji")     # greet → add_emoji（打完招呼后加表情）
graph_builder.add_edge("add_emoji", END)         # add_emoji → 终点（加完表情后结束）

# 4d. 编译图（把图"定型"，使其可以被调用执行）
# ——compile() 之后，graph_builder 变成一个可以直接调用的 graph 对象
# ——之后通过 graph.invoke(初始状态) 来运行整个图
graph = graph_builder.compile()


# ============================================================
# 第五步：运行 Graph！
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("LangGraph 第一课：核心概念")
    print("=" * 50)
    print()

    # 要运行图，需要调用 .invoke() 方法，传入初始状态
    # 初始状态是一个字典，包含图启动时需要的数据
    # 这里我们提供：
    #   - name="小明"（要问候的人）
    #   - greeting=""（空的，等节点来填充）
    initial_state = {"name": "小明", "greeting": ""}

    print("初始状态:", initial_state)
    print()
    print("正在运行图...")
    print("-" * 30)

    # invoke() 会从 START 开始，按照边的顺序依次执行每个节点，直到 END
    # 执行过程：START → greet(生成问候语) → add_emoji(加表情) → END
    # 它返回最终的完整 State（包含所有字段的最终值）
    result = graph.invoke(initial_state)

    print("-" * 30)
    print()
    print("最终状态:", result)
    print()
    print(f"结果: {result['greeting']}")
    print()

    # ========================================================
    # 换一个名字再运行一次，验证图是可以复用的
    # ========================================================
    print("=" * 50)
    print("换一个名字再运行一次...")
    print("=" * 50)
    print()

    # 每次 invoke() 都是独立运行的，互不影响
    # 这次传入 name="小红"
    result2 = graph.invoke({"name": "小红", "greeting": ""})
    print()
    print(f"结果: {result2['greeting']}")

    # ========================================================
    # 核心要点总结
    # ========================================================
    print()
    print("=" * 50)
    print("核心要点：")
    print("=" * 50)
    print("""
    1. State（状态）—— GreetingState
       - 定义图中流转的数据结构（有哪些字段、什么类型）
       - 就像一张有固定字段的表格/共享记事本
       - 用 TypedDict 来定义

    2. Node（节点）—— greet, add_emoji
       - 一个处理数据的 Python 函数
       - 接收 State 作为输入，返回要更新的字段
       - 只返回修改的部分，其他字段自动保持不变

    3. Edge（边）—— add_edge()
       - 定义节点的执行顺序（谁先执行，谁后执行）
       - START → greet → add_emoji → END
       - START 和 END 是 LangGraph 内置的特殊标记

    4. StateGraph（图）
       - 把节点和边组合在一起的容器
       - .compile() 编译图，使其可以运行
       - .invoke(初始状态) 执行整个图，返回最终结果
    """)

    # ========================================================
    # 进阶部分：深入理解 LangGraph 的核心机制
    # ========================================================
    print()
    print("=" * 60)
    print("进阶部分：深入理解核心机制")
    print("=" * 60)

    # ========================================================
    # 进阶 1：Reducer（归约器）—— State 更新的秘密
    # ========================================================
    print("""
    ★ 进阶 1：Reducer（归约器）—— State 更新的底层机制

    当节点返回 {"greeting": "新值"} 时，LangGraph 并不是简单赋值。
    它使用"Reducer"函数来决定如何合并旧值和新值。

    默认 Reducer：直接覆盖（last-write-wins）
        旧值 "hello" + 新值 "world" → 最终 "world"

    自定义 Reducer 示例：
    ─────────────────────────────────────────
    from typing import Annotated
    from operator import add

    class CounterState(TypedDict):
        # 使用 operator.add 作为 Reducer → 累加模式
        count: Annotated[int, add]

    # 节点 A 返回 {"count": 1}
    # 节点 B 返回 {"count": 2}
    # 最终 count = 0 + 1 + 2 = 3（而不是被覆盖为 2）
    ─────────────────────────────────────────

    内置 Reducer:
    • 默认（无 Annotated）：覆盖模式（最后写入的值生效）
    • add_messages：追加消息到列表（第二课会深入学习）
    • operator.add：数值累加 / 列表拼接
    • 自定义函数：def my_reducer(old, new) -> merged
    """)

    # ========================================================
    # 进阶 2：图的可视化 —— 用 Mermaid 查看图结构
    # ========================================================
    print("=" * 60)
    print("★ 进阶 2：图的可视化")
    print("=" * 60)
    print()

    # LangGraph 支持将图导出为 Mermaid 格式，可以在 mermaid.live 在线查看
    try:
        mermaid_code = graph.get_graph().draw_mermaid()
        print("  图的 Mermaid 代码（可在 https://mermaid.live 粘贴查看）：")
        print()
        for line in mermaid_code.split('\n'):
            print(f"    {line}")
        print()
        print("  提示：你也可以用 draw_mermaid_png() 直接生成 PNG 图片")
        print("  （需要安装 pip install pyppeteer）")
    except Exception as e:
        print(f"  图可视化需要 langgraph 0.2+：{e}")
    print()

    # ========================================================
    # 进阶 3：节点返回的多种模式
    # ========================================================
    print("=" * 60)
    print("★ 进阶 3：节点返回值的多种模式")
    print("=" * 60)
    print("""
    节点函数可以返回不同形式的值来更新 State：

    模式 1：部分更新（最常用）
    ─────────────────────────────────────────
    def node(state):
        return {"greeting": "新值"}   # 只更新 greeting
    ─────────────────────────────────────────

    模式 2：不更新任何字段
    ─────────────────────────────────────────
    def log_node(state):
        print(f"当前状态: {state}")
        return {}                     # 返回空字典 = 不修改任何字段
    ─────────────────────────────────────────

    模式 3：同时更新多个字段
    ─────────────────────────────────────────
    def process(state):
        return {
            "greeting": "处理完成",
            "name": state["name"].upper(),  # 同时更新多个字段
        }
    ─────────────────────────────────────────

    重要规则：
    • 返回的字典中的 key 必须是 State 中已定义的字段
    • 未返回的字段会保持原值不变
    • 返回 None 或 {} 都是合法的（不修改任何字段）
    """)

    # ========================================================
    # 进阶 4：图的执行模式 —— invoke vs stream
    # ========================================================
    print("=" * 60)
    print("★ 进阶 4：invoke() vs stream() 两种执行模式")
    print("=" * 60)
    print()

    print("  invoke() —— 一次性返回最终结果（已学）")
    result = graph.invoke({"name": "小张", "greeting": ""})
    print(f"  result = {result}")
    print()

    print("  stream() —— 逐步返回每个节点的输出（流式处理）")
    print("  适合实时展示进度的场景：")
    print()
    for step in graph.stream({"name": "小李", "greeting": ""}):
        # step 是一个字典：{节点名称: 该节点的返回值}
        for node_name, node_output in step.items():
            print(f"    节点 '{node_name}' 输出: {node_output}")
    print()
    print("  stream() 在后续课程中非常重要，尤其是智能体循环中")
    print("  你可以实时看到每一步的执行结果，而不用等全部完成。")

    # ========================================================
    # 进阶总结
    # ========================================================
    print()
    print("=" * 60)
    print("进阶要点总结：")
    print("=" * 60)
    print("""
    1. Reducer（归约器）
       - 控制 State 字段如何合并旧值和新值
       - 默认是覆盖模式，用 Annotated 可切换
       - 常用：add_messages（追加消息）、operator.add（累加）

    2. 图可视化
       - graph.get_graph().draw_mermaid() 生成流程图代码
       - 调试复杂图时非常有用
       - 可以导出为 PNG 图片

    3. 节点返回模式
       - 部分更新、不更新、多字段更新
       - 返回值的 key 必须匹配 State 定义

    4. 两种执行模式
       - invoke()：一次性获取最终结果
       - stream()：逐步获取每个节点的输出
       - stream() 在实时应用中更常用
    """)
