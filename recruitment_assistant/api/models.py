"""
招聘助手 - Pydantic 请求/响应模型（PRD 对齐版）
================================================
"""

from typing import Literal, Optional
from pydantic import BaseModel


# ============================================================
# 请求模型
# ============================================================

class StartRequest(BaseModel):
    """启动招聘流程请求"""
    jd_text: str       # 岗位要求原文
    resume_text: str   # 简历原文


class ResumeRequest(BaseModel):
    """HR 审核决定请求（用于所有 HITL 恢复点）"""
    decision: Literal["approved", "rejected", "offer", "continue"]  # HR 决定
    notes: str = ""           # HR 备注（可选）
    screen_route: Optional[str] = None  # 仅 screening_gate：HR 可覆盖分流决定


class AnswersRequest(BaseModel):
    """录入候选人面试回答"""
    answers: list[str]  # 候选人逐题回答


class FeedbackRequest(BaseModel):
    """面试官反馈提交"""
    interviewer: str    # 面试官姓名
    scores: list[dict]  # [{"question_idx": 0, "score": 8, "comment": "..."}]
    overall: str        # 总体评价


class QuestionsRequest(BaseModel):
    """追加自定义面试题"""
    questions: list[str]  # 新面试题列表


# ============================================================
# 响应模型
# ============================================================

class StartResponse(BaseModel):
    """启动招聘流程响应"""
    session_id: str
    status: str = "running"


class StateResponse(BaseModel):
    """查询当前状态响应"""
    session_id: str
    status: str                             # running / paused / completed / error / rejected
    paused_at: Optional[str] = None         # 暂停时为节点名
    current_step: str = ""
    # JD 解析
    parsed_jd: Optional[dict] = None
    # 简历解析
    parsed_resume: Optional[dict] = None
    # 筛选
    score: Optional[int] = None
    score_reason: Optional[str] = None
    evidence: Optional[list] = None
    screen_route: Optional[str] = None
    # 面试
    interview_questions: Optional[list] = None
    interviewer_feedbacks: Optional[list] = None
    merged_feedback: Optional[dict] = None
    # 评估
    evaluation: Optional[dict] = None
    # 建议
    recommendation: Optional[str] = None
    recommendation_reason: Optional[str] = None
    # Offer
    offer_package: Optional[dict] = None
    # HITL
    hr_action_required: Optional[str] = None
    hr_decisions: Optional[dict] = None


class ReportResponse(BaseModel):
    """获取最终报告响应"""
    session_id: str
    final_report: str
    recommendation: str
    hr_decisions: dict
    offer_package: Optional[dict] = None
