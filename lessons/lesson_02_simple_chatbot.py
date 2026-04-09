"""
=============================================================
第二课：用 LangGraph 构建一个简单的聊天机器人
=============================================================

在第一课中，我们用简单的 Python 函数构建了一个图。
现在我们接入真正的大语言模型（OpenAI GPT）来构建一个聊天机器人！

本课新概念：
- Annotated + add_messages：让状态字段支持"追加模式"（消息越来越多）
- ChatOpenAI：LangChain 对 OpenAI API 的封装
- 消息类型：HumanMessage（用户消息）、AIMessage（AI 回复）

图的结构非常简单：
    [START 起点] --> [chatbot 聊天机器人] --> [END 终点]

虽然结构简单，但 chatbot 节点内部调用了真正的大语言模型！

需要：在 .env 文件中设置 OPENAI_API_KEY
=============================================================
"""

# ============================================================
# 导入需要的模块
# ============================================================

import os  # 用于读取环境变量（API Key）

# Annotated 是 Python 的类型标注工具
# 它可以给类型附加额外信息。在 LangGraph 中，我们用它来标记
# "这个字段应该用追加模式，而不是覆盖模式"
from typing import Annotated

# TypedDict：定义有固定字段的字典类型（第一课已学过）
from typing_extensions import TypedDict

# load_dotenv：从 .env 文件中加载环境变量（比如 API Key）
# 这样你不需要把敏感信息写在代码里
from dotenv import load_dotenv

# LangGraph 的核心组件（第一课已学过）
from langgraph.graph import StateGraph, START, END

# add_messages 是 LangGraph 提供的一个特殊函数
# 当它和 Annotated 一起使用时，表示"这个列表字段使用追加模式"
# 即：新消息会被追加到列表末尾，而不是替换整个列表
from langgraph.graph.message import add_messages

# ChatOpenAI 是 LangChain 对 OpenAI 聊天模型的封装
# 它帮我们处理 API 调用的细节（请求格式、认证、重试等）
from langchain_openai import ChatOpenAI

# 消息类型：
# HumanMessage —— 代表用户发送的消息
# AIMessage —— 代表 AI（大语言模型）的回复
from langchain_core.messages import HumanMessage, AIMessage

# 从 .env 文件加载环境变量
# 这会读取项目根目录下的 .env 文件，把里面的键值对设置为环境变量
load_dotenv()


# ============================================================
# 新概念：Annotated 状态与 add_messages
# ============================================================
#
# 回顾第一课：
#   当节点返回 {"greeting": "新值"} 时，它会【覆盖】旧值。
#   旧值被新值完全替换了。
#
# 但对于聊天机器人，我们需要的是【追加】行为：
#   每次对话都会产生新消息，新消息应该被追加到消息列表中，
#   而不是替换掉之前的所有消息。否则就丢失了对话历史！
#
# Annotated[list, add_messages] 告诉 LangGraph：
#   "当有人返回新消息时，把它们【追加】到列表末尾，
#    而不是替换整个列表。"
#
# 对比：
#   普通字段：   state["greeting"] = "新值"          → 直接覆盖
#   Annotated：  state["messages"].append(新消息)     → 追加到末尾
#
# 这是 LangGraph 中非常重要的概念！几乎所有聊天相关的应用都会用到它。

class ChatState(TypedDict):
    """
    聊天状态：包含一个不断增长的消息列表。

    messages 字段使用 Annotated[list, add_messages]：
    - 新消息会被追加到列表末尾，而不是替换整个列表
    - 这样对话历史就能完整保留下来
    """
    messages: Annotated[list, add_messages]  # 消息列表（追加模式）


# ============================================================
# 创建大语言模型（LLM）实例
# ============================================================
# ChatOpenAI 是 LangChain 对 OpenAI API 的封装类。
# 它帮我们处理所有 API 调用的细节，我们只需要调用 .invoke() 即可。
#
# 参数说明：
# - model：使用的模型名称，从环境变量 DEFAULT_LLM_PROVIDER 读取
#   这样你可以在 .env 文件中统一修改，不用改代码
# - temperature=0.7：控制回复的随机性/创造性
#   0 = 非常确定/稳定（每次回复几乎一样）
#   1 = 非常有创意/随机（每次回复差异较大）

# 从环境变量读取模型名称和 API 地址
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_PROVIDER", "gpt-4o-mini")
BASE_URL = os.getenv("BASE_URL") or None  # 为空则使用 OpenAI 官方地址

llm = ChatOpenAI(
    model=DEFAULT_MODEL,      # 模型名称（从环境变量读取）
    base_url=BASE_URL,        # API 地址（从环境变量读取，为空则用官方地址）
    temperature=0.7,          # 创造性程度：0=稳定，1=随机
)


# ============================================================
# 定义聊天机器人节点
# ============================================================
def chatbot(state: ChatState) -> dict:
    """
    聊天机器人节点：
    1. 从 State 中取出所有消息（对话历史）
    2. 把消息发送给大语言模型（LLM）
    3. LLM 生成回复
    4. 返回回复（会被【追加】到 messages 列表中）
    """
    # state["messages"] 包含完整的对话历史
    # LLM 会读取所有历史消息来理解上下文，然后生成回复
    response = llm.invoke(state["messages"])

    # 返回 LLM 的回复
    # 因为 messages 字段使用了 Annotated[list, add_messages]，
    # 所以这条回复会被【追加】到消息列表末尾，而不是替换
    return {"messages": [response]}


# ============================================================
# 构建图
# ============================================================
# 和第一课一样的步骤：创建图 → 添加节点 → 添加边 → 编译
graph_builder = StateGraph(ChatState)

# 添加聊天机器人节点
graph_builder.add_node("chatbot", chatbot)

# 简单的流程：起点 → chatbot → 终点
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 编译图
graph = graph_builder.compile()


# ============================================================
# 运行！
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("LangGraph 第二课：简单聊天机器人")
    print("=" * 50)
    print()

    # 检查 API Key 是否已设置
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-your"):
        print("错误：请先设置你的 OpenAI API Key！")
        print("  1. 把 .env.example 复制为 .env")
        print("  2. 把 'sk-your-api-key-here' 替换成你的真实 API Key")
        print("  获取 Key：https://platform.openai.com/api-keys")
        exit(1)

    # ========================================================
    # 示例 1：发送单条消息
    # ========================================================
    print("--- 示例 1：发送单条消息 ---")
    print()

    # 用 HumanMessage 包装用户的消息，传入图中
    result = graph.invoke({
        "messages": [HumanMessage(content="What is LangGraph in one sentence?")]
    })

    # 结果中包含所有消息（用户输入 + LLM 回复）
    for msg in result["messages"]:
        # 判断消息类型，显示对应的角色标签
        role = "你" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content}")

    print()

    # ========================================================
    # 示例 2：多轮对话
    # ========================================================
    # 我们可以预填对话历史！LLM 会读取所有历史消息来理解上下文。
    print("--- 示例 2：多轮对话 ---")
    print()

    result2 = graph.invoke({
        "messages": [
            HumanMessage(content="My name is Alice."),                              # 第1轮：用户自我介绍
            AIMessage(content="Nice to meet you, Alice! How can I help you today?"),# 第1轮：AI 回复
            HumanMessage(content="What's my name?"),                                # 第2轮：用户追问
        ]
    })

    # 打印最后一条消息（LLM 的最新回复）
    last_message = result2["messages"][-1]
    print(f"  AI 记住了: {last_message.content}")

    print()

    # ========================================================
    # 示例 3：交互式聊天循环
    # ========================================================
    print("--- 示例 3：交互式聊天 ---")
    print("  输入 'quit' 退出")
    print()

    # 手动维护对话历史列表
    # 每次循环都把新消息加进去，这样 LLM 就能看到完整的对话上下文
    conversation = []

    while True:
        user_input = input("  你: ")
        if user_input.lower() in ("quit", "exit", "q"):
            print("  再见！")
            break

        # 把用户的输入包装成 HumanMessage，加入对话历史
        conversation.append(HumanMessage(content=user_input))

        # 把完整的对话历史传给图，让 LLM 生成回复
        result = graph.invoke({"messages": conversation})

        # 获取 AI 的回复（消息列表中的最后一条）
        ai_message = result["messages"][-1]
        print(f"  AI: {ai_message.content}")
        print()

        # 用完整的结果更新对话历史
        # 这样下一轮对话就包含了所有之前的消息
        conversation = result["messages"]

    # ========================================================
    # 核心要点总结
    # ========================================================
    print()
    print("=" * 50)
    print("核心要点：")
    print("=" * 50)
    print("""
    1. Annotated[list, add_messages]
       - 让状态字段使用"追加模式"
       - 新消息会被追加到列表末尾，而不是替换整个列表
       - 这是聊天应用中最重要的状态模式

    2. ChatOpenAI
       - LangChain 对 OpenAI 模型的封装
       - llm.invoke(messages) 发送消息并获取回复
       - temperature 控制回复的随机性/创造性

    3. 消息类型
       - HumanMessage：用户发送的消息
       - AIMessage：AI（LLM）的回复
       - 消息类型帮助 LLM 区分"谁说了什么"

    4. 对话历史
       - LLM 会看到 State 中的所有消息
       - 这就是它"记住"对话内容的方式
       - 消息越多 = 上下文越丰富（但也越费钱）
    """)

    # ========================================================
    # 进阶部分：更深入的聊天技巧
    # ========================================================
    print()
    print("=" * 60)
    print("进阶部分：更深入的聊天技巧")
    print("=" * 60)

    # ========================================================
    # 进阶 1：stream() 流式输出 —— 实时看到 AI 的回答
    # ========================================================
    print("""
    ★ 进阶 1：stream() 流式输出 —— 实时看到 AI 打字

    invoke() 会等 AI 回复完整后才返回（可能要等几秒）。
    stream() 可以逐步输出，让用户像看 AI 打字一样实时看到。

    LangGraph 有两种 stream 模式：
    • graph.stream()       —— 逐节点输出（节点级别）
    • graph.astream_events() —— 逐 token 输出（字符级别，需要异步）
    """)

    print("--- 流式输出演示（节点级别）---")
    print()
    for step in graph.stream({
        "messages": [HumanMessage(content="Tell me a joke in one sentence.")]
    }):
        for node_name, output in step.items():
            last_msg = output["messages"][-1]
            print(f"  [{node_name}] {last_msg.content[:100]}")
    print()

    # ========================================================
    # 进阶 2：SystemMessage —— 给 AI 设定"人设"
    # ========================================================
    from langchain_core.messages import SystemMessage

    print("=" * 60)
    print("★ 进阶 2：SystemMessage —— 给 AI 设定角色人设")
    print("=" * 60)
    print("""
    SystemMessage 告诉 LLM "你是什么角色"、"你应该怎么回答"。
    它通常放在消息列表的最前面，LLM 会遵循这个设定。
    """)

    # 不改变图结构，通过初始消息传入 SystemMessage
    result = graph.invoke({
        "messages": [
            SystemMessage(content=(
                "你是一个海盗风格的 AI 助手。"
                "你的每句回答都要用海盗的语气，"
                "经常说'呀嗬''宝藏''船长'等词。"
                "保持有趣但有用。"
            )),
            HumanMessage(content="What is Python?"),
        ]
    })
    print(f"  海盗 AI: {result['messages'][-1].content}")
    print()

    print("  提示：SystemMessage 非常强大，可以用来：")
    print("    - 设定回复语言（中文/英文/日语）")
    print("    - 限制回复长度（'回答不超过 50 字'）")
    print("    - 设定专业领域（'你是一个 Python 专家'）")
    print("    - 定义输出格式（'用 JSON 格式回答'）")
    print()

    # ========================================================
    # 进阶 3：add_messages 的深层理解
    # ========================================================
    print("=" * 60)
    print("★ 进阶 3：add_messages Reducer 的深层理解")
    print("=" * 60)
    print("""
    add_messages 不仅仅是"追加"，它还支持：

    1. 追加新消息（最常用）
       ─────────────────────────────────────────
       return {"messages": [new_msg]}
       # 结果：旧消息列表 + new_msg
       ─────────────────────────────────────────

    2. 按 ID 更新已有消息
       ─────────────────────────────────────────
       # 如果返回的消息 ID 与已有消息相同，会覆盖而非追加
       updated_msg = AIMessage(content="更正内容", id=old_msg.id)
       return {"messages": [updated_msg]}
       ─────────────────────────────────────────

    3. 删除消息（用 RemoveMessage）
       ─────────────────────────────────────────
       from langchain_core.messages import RemoveMessage
       return {"messages": [RemoveMessage(id=msg_to_delete.id)]}
       ─────────────────────────────────────────

    这些高级操作在管理对话历史长度时非常有用！
    比如当消息太多导致 Token 超限时，可以删除早期消息。
    """)

    # ========================================================
    # 进阶 4：Token 消耗与成本意识
    # ========================================================
    print("=" * 60)
    print("★ 进阶 4：Token 消耗与成本管理")
    print("=" * 60)
    print("""
    每次调用 LLM 都要花钱！理解 Token 消耗很重要：

    Token 计算规则（粗略估算）：
    • 1 个英文单词 ≈ 1-2 tokens
    • 1 个中文字符 ≈ 1-2 tokens
    • 代码通常比自然语言消耗更多 tokens

    对话历史是最大的 Token 消耗源：
    ┌─────────────────────────────────────────────┐
    │ 第 1 轮: 用户(10) + AI(50) = 60 tokens      │
    │ 第 2 轮: 历史(60) + 用户(10) + AI(50) = 120 │
    │ 第 3 轮: 历史(120) + 用户(10) + AI(50) = 180│
    │ ...消耗线性增长...                            │
    └─────────────────────────────────────────────┘

    控制成本的策略：
    1. 限制对话历史长度（只保留最近 N 轮）
    2. 使用 RemoveMessage 删除早期消息
    3. 用摘要替代完整历史（"前 10 轮总结为..."）
    4. 选择更便宜的模型（gpt-4o-mini vs gpt-4o）
    5. 降低 temperature 减少重试需求

    模型价格对比（参考，可能有变化）：
    • gpt-4o-mini: ~$0.15/1M input tokens
    • gpt-4o:      ~$2.50/1M input tokens
    • gpt-4.1:     ~$2.00/1M input tokens
    """)

    # ========================================================
    # 进阶总结
    # ========================================================
    print("=" * 60)
    print("进阶要点总结：")
    print("=" * 60)
    print("""
    1. stream() 流式输出
       - graph.stream() 逐节点输出
       - 用于实时展示 AI 回复过程
       - 用户体验更好（不用干等）

    2. SystemMessage 角色人设
       - 放在消息列表最前面
       - 控制 AI 的回复风格、语言、格式
       - 是构建专业 AI 助手的关键

    3. add_messages 高级操作
       - 追加、按 ID 更新、删除消息
       - RemoveMessage 管理历史长度
       - 解决 Token 超限问题

    4. Token 与成本管理
       - 对话历史是最大开销
       - 限制历史长度、用摘要替代
       - 选择合适的模型平衡成本和质量
    """)
