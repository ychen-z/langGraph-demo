"""
招聘助手 - 共享状态定义
========================
所有智能体节点共享的数据结构，贯穿整个招聘流水线。

数据流（PRD 对齐）：
  parse_jd       写入 parsed_jd
  → resume_parser  写入 parsed_resume
  → screener       写入 score, score_reason, evidence
  → screening_gate 读取 hr_decisions, 写入 screen_route（4 路分流）
  → interview_generator 写入 interview_questions（含 rubrics）
  → interview_gate      读取 hr_decisions（pass-through）
  → collect_feedback     读取 interviewer_feedbacks（HITL，多面试官）
  → merge_feedback       写入 merged_feedback（合并分歧）
  → decision_gate        读取 hr_decisions（人工确认 continue/reject/offer）
  → offer_pack           写入 offer_package（仅 offer 路径）
  → report_generator     写入 final_report, status
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class RecruitmentState(TypedDict):
    """
    招聘流程的共享状态。

    对齐 PRD 8 节点架构：
      parse_jd → screen_resume → route_screen → generate_questions
      → collect_feedback → merge_feedback → decision_gate → offer_pack
    """

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
                              #    "follow_up_suggestions": [str]}

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
    hr_decisions: dict         # {"screening": "approved"|"rejected",
                              #  "screen_route": "phone_screen"|"onsite"|...,
                              #  "interview": "approved"|"rejected",
                              #  "final": "offer"|"rejected"|"continue"}
    messages: Annotated[list, add_messages]  # 消息历史（追加模式）
