"""
招聘助手 - 图管线（PRD 对齐版）
=================================
构建完整的招聘流程图，包含 4 路分流和 HITL 暂停点。

完整流程图：
  START → parse_jd → resume_parser → screener → [HITL#1] → screening_gate
    → (reject)           → report_generator → END
    → (need_more_info)   → report_generator → END  (暂时同 reject，可扩展)
    → (phone_screen)     → interview_generator → [HITL#2] → interview_gate
    → (onsite)           → interview_generator → [HITL#2] → interview_gate
        → (rejected)     → report_generator → END
        → (approved)     → evaluator → [HITL#3] → collect_feedback
                         → merge_feedback → advisor → [HITL#4] → decision_gate
                            → (rejected)  → report_generator → END
                            → (offer)     → offer_pack → report_generator → END
                            → (continue)  → report_generator → END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import RecruitmentState
from .agents import (
    parse_jd,
    resume_parser,
    screener,
    screening_gate,
    interview_generator,
    interview_gate,
    evaluator,
    collect_feedback,
    merge_feedback,
    advisor,
    decision_gate,
    offer_pack,
    report_generator,
)


# ============================================================
# 路由函数
# ============================================================

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
    """
    面试后路由。HR 通过 hr_decisions.interview 控制。
    """
    hr = state.get("hr_decisions", {})
    if hr.get("interview") == "rejected":
        return "report_generator"
    return "evaluator"


def route_after_decision(state: RecruitmentState) -> str:
    """
    最终决策后路由（PRD: decision_gate 出口）。
    HR 通过 hr_decisions.final 控制。
    - offer    → offer_pack → report_generator
    - rejected → report_generator
    - continue → report_generator（可扩展为补面循环）
    """
    hr = state.get("hr_decisions", {})
    final = hr.get("final", "continue")

    if final == "offer":
        return "offer_pack"
    else:
        # rejected 和 continue 都直接到报告
        return "report_generator"


# ============================================================
# 构建图
# ============================================================

def build_graph(checkpointer=None):
    """
    构建并编译招聘流程图。

    HITL 暂停点（interrupt_before）：
    1. screening_gate   — HR 审核筛选结果，决定分流
    2. interview_gate   — HR 审核面试题，录入候选人回答
    3. collect_feedback  — 等待多面试官提交反馈
    4. decision_gate    — HR 做最终录用决策
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(RecruitmentState)

    # ---- 注册所有节点 ----
    graph.add_node("parse_jd", parse_jd)
    graph.add_node("resume_parser", resume_parser)
    graph.add_node("screener", screener)
    graph.add_node("screening_gate", screening_gate)
    graph.add_node("interview_generator", interview_generator)
    graph.add_node("interview_gate", interview_gate)
    graph.add_node("evaluator", evaluator)
    graph.add_node("collect_feedback", collect_feedback)
    graph.add_node("merge_feedback", merge_feedback)
    graph.add_node("advisor", advisor)
    graph.add_node("decision_gate", decision_gate)
    graph.add_node("offer_pack", offer_pack)
    graph.add_node("report_generator", report_generator)

    # ---- 普通边（固定流向）----
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

    # ---- 条件边（动态路由）----
    # 筛选后 4 路分流
    graph.add_conditional_edges("screening_gate", route_after_screening)
    # 面试后路由
    graph.add_conditional_edges("interview_gate", route_after_interview)
    # 最终决策路由
    graph.add_conditional_edges("decision_gate", route_after_decision)

    # ---- 编译（带 HITL 暂停点）----
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "screening_gate",    # HITL #1: 筛选审核
            "interview_gate",    # HITL #2: 面试审核
            "collect_feedback",  # HITL #3: 多面试官反馈收集
            "decision_gate",     # HITL #4: 最终决策
        ],
    )


# ============================================================
# 模块级实例（单例）
# ============================================================
_memory = MemorySaver()
recruitment_graph = build_graph(checkpointer=_memory)
