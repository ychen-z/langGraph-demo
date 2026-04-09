"""
=============================================================
第五课：检查点与记忆（Checkpointing & Memory）
=============================================================

问题回顾：
在第二课中，我们的聊天机器人可以对话，但有两个致命缺陷：
1. 程序重启后，对话记录全部丢失（没有持久化）
2. 无法同时处理多个用户的对话（没有隔离）

解决方案：检查点（Checkpointing），也叫持久化/记忆

LangGraph 可以在每一步执行后保存状态，这样你就能：
- 以后恢复之前的对话（断点续传）
- 用不同的"线程"处理多个用户（互不干扰）
- 回溯到之前的某个状态（时间旅行）

本课新概念：
- MemorySaver：内存检查点（把状态存在 Python 内存中，开发调试用）
- thread_id：线程 ID，标识一个独立的对话
- config：传给 invoke() 的配置，指定使用哪个线程

架构示意：
    [用户A, thread_1] --> 图 --> [状态保存到检查点]
    [用户B, thread_2] --> 图 --> [状态分别保存]

    过了一段时间...
    [用户A, thread_1] --> 图 --> [自动加载之前的状态] --> 继续对话！

需要：在 .env 文件中设置 OPENAI_API_KEY
=============================================================
"""

# ============================================================
# 导入模块
# ============================================================

import os                            # 读取环境变量
from typing import Annotated         # 类型标注（追加模式）
from typing_extensions import TypedDict  # 定义状态结构
from dotenv import load_dotenv       # 加载 .env 文件

from langgraph.graph import StateGraph, START, END  # 图的核心组件
from langgraph.graph.message import add_messages    # 消息追加模式

# ★ 本课重点：MemorySaver —— 内存检查点
# 它把图的状态保存在 Python 的内存（RAM）中
# 优点：简单快速，适合开发和调试
# 缺点：程序关闭后数据就丢了
# 生产环境应该用 SqliteSaver 或 PostgresSaver
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI              # OpenAI LLM 封装
from langchain_core.messages import HumanMessage     # 用户消息类型

# 加载环境变量
load_dotenv()


# ============================================================
# 第一步：定义状态和图（和第二课一样）
# ============================================================

class ChatState(TypedDict):
    """聊天状态：包含不断增长的消息列表。"""
    messages: Annotated[list, add_messages]  # 消息列表（追加模式）


# 从环境变量读取默认模型名称和 API 地址
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_PROVIDER", "gpt-4o-mini")
BASE_URL = os.getenv("BASE_URL") or None  # 为空则使用 OpenAI 官方地址

# 创建 LLM 实例
llm = ChatOpenAI(model=DEFAULT_MODEL, base_url=BASE_URL, temperature=0.7)


def chatbot(state: ChatState) -> dict:
    """聊天机器人节点：调用 LLM 生成回复。"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 构建图（和第二课完全一样）
graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# ============================================================
# 第二步：添加检查点（Checkpointer）
# ============================================================
#
# ★ 和第二课的唯一区别就在这里！
# 我们在 compile() 时传入了一个 checkpointer 参数。
# 这告诉 LangGraph："每次执行完一步，都把状态保存起来。"
#
# MemorySaver 把状态存在 Python 字典中（内存里）。
# 生产环境你应该换成：
# - SqliteSaver：存在本地 SQLite 数据库（单服务器）
# - PostgresSaver：存在 PostgreSQL 数据库（多服务器，生产推荐）

memory = MemorySaver()

# ★ 关键区别：compile 时传入 checkpointer
# 对比第二课：graph = graph_builder.compile()          （无记忆）
# 本课：       graph = graph_builder.compile(checkpointer=memory)（有记忆）
graph = graph_builder.compile(checkpointer=memory)


# ============================================================
# 第三步：使用 thread_id 管理对话
# ============================================================
#
# 什么是 thread_id？
# ——可以把它想象成"聊天室的房间号"。
#   不同的 thread_id 代表不同的对话，互相隔离。
#   同一个 thread_id 的消息会自动串联起来。
#
# 怎么传 thread_id？
# ——通过 config 参数：config={"configurable": {"thread_id": "房间号"}}

if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 第五课：检查点与记忆")
    print("=" * 60)
    print()

    # 检查 API Key
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-your"):
        print("错误：请在 .env 文件中设置 OPENAI_API_KEY")
        exit(1)

    # ========================================================
    # 演示 1：带记忆的对话
    # ========================================================
    print("--- 演示 1：带记忆的对话 ---")
    print()

    # config 指定使用哪个对话线程
    # 把 thread_id 想象成"聊天室房间号"
    config_alice = {"configurable": {"thread_id": "alice_chat_1"}}

    # 第 1 轮：Alice 自我介绍
    print("第 1 轮:")
    result = graph.invoke(
        {"messages": [HumanMessage(content="Hi! My name is Alice. I'm learning Python.")]},
        config=config_alice,  # ← 使用 Alice 的线程
    )
    print(f"  Alice: Hi! My name is Alice. I'm learning Python.")
    print(f"  AI: {result['messages'][-1].content}")
    print()

    # 第 2 轮：Alice 追问
    # 注意：我们不需要传完整的对话历史！
    # 检查点会自动加载之前的消息。我们只需要传新消息。
    print("第 2 轮:")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What's my name? And what am I learning?")]},
        config=config_alice,  # ← 同一个线程 = 同一个对话（自动加载历史）
    )
    print(f"  Alice: What's my name? And what am I learning?")
    print(f"  AI: {result['messages'][-1].content}")
    print()

    # ========================================================
    # 演示 2：不同线程隔离不同用户
    # ========================================================
    print("--- 演示 2：不同用户（不同线程）---")
    print()

    # Bob 使用自己的线程，和 Alice 完全隔离
    config_bob = {"configurable": {"thread_id": "bob_chat_1"}}

    # Bob 的对话是完全独立的
    print("Bob 的线程:")
    result = graph.invoke(
        {"messages": [HumanMessage(content="Hi! I'm Bob. I love cooking.")]},
        config=config_bob,  # ← Bob 的线程
    )
    print(f"  Bob: Hi! I'm Bob. I love cooking.")
    print(f"  AI: {result['messages'][-1].content}")
    print()

    # Bob 追问自己的爱好
    result = graph.invoke(
        {"messages": [HumanMessage(content="What's my hobby?")]},
        config=config_bob,  # ← 同一个线程
    )
    print(f"  Bob: What's my hobby?")
    print(f"  AI: {result['messages'][-1].content}")
    print()

    # ========================================================
    # 演示 3：证明 Alice 的线程仍然完好
    # ========================================================
    print("--- 演示 3：Alice 的线程仍然正常 ---")
    print()

    # 切回 Alice 的线程，之前的对话历史应该还在
    result = graph.invoke(
        {"messages": [HumanMessage(content="Remind me, what was I learning?")]},
        config=config_alice,  # ← 回到 Alice 的线程
    )
    print(f"  Alice: Remind me, what was I learning?")
    print(f"  AI: {result['messages'][-1].content}")
    print()

    # ========================================================
    # 演示 4：查看保存的状态
    # ========================================================
    print("--- 演示 4：查看保存的状态 ---")
    print()

    # 你可以通过 get_state() 查看任意线程的当前状态
    saved_state = graph.get_state(config_alice)
    msg_count = len(saved_state.values["messages"])
    print(f"  Alice 的线程中有 {msg_count} 条消息")
    print(f"  消息列表:")
    for i, msg in enumerate(saved_state.values["messages"]):
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        # 如果消息太长就截断显示
        content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"    {i+1}. [{role}] {content}")

    # ========================================================
    # 核心要点总结
    # ========================================================
    print()
    print("=" * 60)
    print("核心要点：")
    print("=" * 60)
    print("""
    1. MemorySaver（内存检查点）
       - 把图的状态存在内存中
       - 用法：compile(checkpointer=memory)
       - 仅限开发调试，重启程序后数据丢失
       - 生产环境用 SqliteSaver 或 PostgresSaver

    2. thread_id（线程 ID）
       - 标识一个独立的对话（像"聊天室房间号"）
       - 用法：config={"configurable": {"thread_id": "..."}}
       - 不同 thread_id = 不同的对话，互不干扰

    3. 自动历史管理
       - 你只需要传入新消息
       - 检查点会自动加载之前的消息历史
       - 不需要再手动维护对话历史列表了！

    4. 多用户支持
       - 每个用户分配一个唯一的 thread_id
       - 对话之间完全隔离
       - 这就是真实聊天产品处理多用户的方式

    5. 生产环境的持久化选择
       - MemorySaver：仅开发用（重启即丢失）
       - SqliteSaver：单服务器部署
       - PostgresSaver：多服务器部署，生产推荐
    """)

    # ========================================================
    # 进阶部分：检查点的高级用法
    # ========================================================
    print()
    print("=" * 60)
    print("进阶部分：检查点的高级用法")
    print("=" * 60)

    # ========================================================
    # 进阶 1：状态历史回溯（时间旅行）
    # ========================================================
    print("""
    ★ 进阶 1：状态历史回溯 —— 时间旅行（Time Travel）

    LangGraph 保存的不仅是"最新状态"，而是每一步的状态快照！
    你可以回到过去的任意一步，查看当时的状态。

    核心方法：graph.get_state_history(config)
    它返回一个迭代器，从最新状态到最早状态。
    """)

    print("  Alice 线程的状态历史：")
    print()

    history = list(graph.get_state_history(config_alice))
    for i, snapshot in enumerate(history[:5]):  # 只显示最近 5 个快照
        msg_count = len(snapshot.values.get("messages", []))
        # snapshot.metadata 包含执行信息
        step = snapshot.metadata.get("step", "?")
        source = snapshot.metadata.get("source", "?")
        print(f"    快照 {i}: step={step}, source={source}, "
              f"消息数={msg_count}")

    print()
    print("""
    每个快照包含：
    • values：当时的完整 State（所有字段的值）
    • metadata：元信息（步骤号、来源节点等）
    • config：该快照的配置（包含 checkpoint_id）

    你可以用快照的 config 来"回到过去"：
    ─────────────────────────────────────────
    # 获取某个历史快照
    old_snapshot = history[3]

    # 从那个时间点继续执行（分支出新的对话线）
    result = graph.invoke(
        {"messages": [HumanMessage(content="新消息")]},
        config=old_snapshot.config,
    )
    ─────────────────────────────────────────
    """)

    # ========================================================
    # 进阶 2：update_state —— 手动修改状态
    # ========================================================
    print("=" * 60)
    print("★ 进阶 2：update_state() —— 手动修改保存的状态")
    print("=" * 60)
    print("""
    有时你想在恢复执行前修改状态（比如纠正 AI 的错误）。

    ─────────────────────────────────────────
    # 查看当前状态
    state = graph.get_state(config)

    # 手动修改状态中的消息
    graph.update_state(
        config,
        {"messages": [HumanMessage(content="纠正后的消息")]},
    )

    # 从修改后的状态继续执行
    result = graph.invoke(None, config)
    ──────────────────────────────────���──────

    update_state 的高级用法 —— 指定 as_node：
    ─────────────────────────────────────────
    # 假装这个更新来自某个节点
    graph.update_state(
        config,
        {"messages": [AIMessage(content="修正后的回复")]},
        as_node="chatbot",  # 假装是 chatbot 节点的输出
    )
    ─────────────────────────────────────────

    这在第六课（人机协作）中非常有用，
    用来注入"拒绝"消息或修正 AI 的决策。
    """)

    # ========================================================
    # 进阶 3：检查点的性能考量
    # ========================================================
    print("=" * 60)
    print("★ 进阶 3：检查点的性能与存储考量")
    print("=" * 60)
    print("""
    检查点会保存每一步的完整状态快照，需要注意：

    存储增长：
    ┌──────────────────────────────────────────────────┐
    │ 对话 10 轮 × 每轮 2 个快照(input+output)         │
    │ = 20 个快照 × 每个约 2KB                          │
    │ = ~40KB / 每个线程                                │
    │                                                  │
    │ 1000 个并发用户 × 40KB = ~40MB                    │
    │ 生产环境可能需要定期清理旧快照                      │
    └──────────────────────────────────────────────────┘

    生产环境的最佳实践：
    1. 使用 PostgresSaver 配合连接池
    2. 设置 TTL（过期时间）自动清理旧数据
    3. 只在需要时启用检查点（简单的一次性图不需要）
    4. 消息列表中避免存储大型二进制数据

    检查点 vs 外部数据库：
    • 检查点用于保存"图的执行状态"（消息、中间结果）
    • 业务数据（用户信息、订单等）应该存在外部数据库
    • 不要把检查点当作通用数据库使用
    """)

    # ========================================================
    # 进阶总结
    # ========================================================
    print("=" * 60)
    print("进阶要点总结：")
    print("=" * 60)
    print("""
    1. 状态历史回溯（时间旅行）
       - get_state_history(config) 获取所有历史快照
       - 可以回到任意历史节点继续执行
       - 适合调试和分析执行过程

    2. update_state() 手动修改状态
       - 在恢复执行前纠正 AI 的错误
       - as_node 参数可以假装来自某个节点
       - 第六课 HITL 的核心机制

    3. 性能考量
       - 检查点会保存每步快照，注意存储增长
       - 生产环境使用 PostgresSaver + 连接池
       - 定期清理旧数据，设置合理的 TTL
       - 检查点 ≠ 通用数据库
    """)
