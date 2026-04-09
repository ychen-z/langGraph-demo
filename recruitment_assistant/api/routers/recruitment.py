"""
招聘助手 - 招聘流程路由（PRD 对齐版）
======================================
支持 SSE 实时推送、多面试官反馈、4路分流、Offer 生成。
"""

import uuid
import json
import asyncio
import queue
import threading
from collections import defaultdict
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from ..models import (
    StartRequest, StartResponse,
    ResumeRequest, AnswersRequest, FeedbackRequest, QuestionsRequest,
    StateResponse, ReportResponse,
)
from ...graph.pipeline import recruitment_graph
from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/api/recruitment", tags=["recruitment"])

# ============================================================
# HITL 提示文案（PRD 对齐）
# ============================================================
HR_ACTION_MAP = {
    "screening_gate": "请审核筛选结果（评分+证据引用），选择分流方向（直接面试/电话面/淘汰/补充信息）",
    "interview_gate": "请审核面试题及评分标准，并录入候选人回答后继续",
    "collect_feedback": "请等待面试官提交反馈，或直接继续（将使用自动评估结果）",
    "decision_gate": "请审核录用建议和面评分析，做出最终决策（Offer/淘汰/补面）",
}

# 节点中文名映射（用于 SSE 事件展示）
NODE_LABEL = {
    "parse_jd": "JD解析",
    "resume_parser": "简历解析",
    "screener": "筛选评分",
    "screening_gate": "筛选审核",
    "interview_generator": "面试题生成",
    "interview_gate": "面试审核",
    "evaluator": "面试评估",
    "collect_feedback": "面试官反馈",
    "merge_feedback": "面评合并",
    "advisor": "录用建议",
    "decision_gate": "最终决策",
    "offer_pack": "Offer生成",
    "report_generator": "报告生成",
}

# ============================================================
# SSE 事件总线
# ============================================================
_sse_subscribers: dict[str, list[queue.Queue]] = defaultdict(list)
_running_sessions: set[str] = set()
_lock = threading.Lock()


def emit_sse(session_id: str, event_type: str, data: dict):
    """向该 session 的所有 SSE 订阅者推送一个事件。"""
    for q in _sse_subscribers.get(session_id, []):
        try:
            q.put_nowait({"event": event_type, "data": data})
        except queue.Full:
            pass


# ============================================================
# 后台图执行（使用 stream 逐节点输出）
# ============================================================
def _run_graph_streamed(session_id: str, input_data: dict | None, config: dict):
    """后台线程：逐节点执行并通过 SSE 推送事件。"""
    try:
        emit_sse(session_id, "status", {"status": "running", "message": "流程执行中..."})

        for chunk in recruitment_graph.stream(input_data, config=config):
            for node_name, node_output in chunk.items():
                if node_name.startswith("__"):
                    continue

                label = NODE_LABEL.get(node_name, node_name)
                event_data = {"node": node_name, "label": label}

                # 根据节点附加关键数据
                if isinstance(node_output, dict):
                    if node_name == "parse_jd":
                        event_data["parsed_jd"] = node_output.get("parsed_jd")
                    elif node_name == "resume_parser":
                        event_data["parsed_resume"] = node_output.get("parsed_resume")
                    elif node_name == "screener":
                        event_data["score"] = node_output.get("score")
                        event_data["score_reason"] = node_output.get("score_reason")
                        event_data["evidence"] = node_output.get("evidence")
                        event_data["screen_route"] = node_output.get("screen_route")
                    elif node_name == "interview_generator":
                        event_data["interview_questions"] = node_output.get("interview_questions")
                    elif node_name == "evaluator":
                        event_data["evaluation"] = node_output.get("evaluation")
                    elif node_name == "merge_feedback":
                        event_data["merged_feedback"] = node_output.get("merged_feedback")
                    elif node_name == "advisor":
                        event_data["recommendation"] = node_output.get("recommendation")
                        event_data["recommendation_reason"] = node_output.get("recommendation_reason")
                    elif node_name == "offer_pack":
                        event_data["offer_package"] = node_output.get("offer_package")
                    elif node_name == "report_generator":
                        event_data["final_report"] = node_output.get("final_report")
                        event_data["final_status"] = node_output.get("status")

                emit_sse(session_id, "node_complete", event_data)

        # stream 结束后检查暂停状态
        snapshot = recruitment_graph.get_state(config)
        if snapshot and snapshot.next:
            paused_at = snapshot.next[0]
            values = snapshot.values or {}

            pause_data = {
                "paused_at": paused_at,
                "hr_action": HR_ACTION_MAP.get(paused_at, ""),
                # 通用状态
                "parsed_jd": values.get("parsed_jd"),
                "parsed_resume": values.get("parsed_resume"),
                "score": values.get("score"),
                "score_reason": values.get("score_reason"),
                "evidence": values.get("evidence"),
                "screen_route": values.get("screen_route"),
                "interview_questions": values.get("interview_questions"),
                "interviewer_feedbacks": values.get("interviewer_feedbacks"),
                "merged_feedback": values.get("merged_feedback"),
                "evaluation": values.get("evaluation"),
                "recommendation": values.get("recommendation"),
                "recommendation_reason": values.get("recommendation_reason"),
                "offer_package": values.get("offer_package"),
            }
            emit_sse(session_id, "paused", pause_data)
        else:
            values = snapshot.values if snapshot else {}
            emit_sse(session_id, "finished", {
                "status": values.get("status", "completed"),
                "final_report": values.get("final_report", ""),
                "recommendation": values.get("recommendation", ""),
                "hr_decisions": values.get("hr_decisions", {}),
                "offer_package": values.get("offer_package"),
            })

    except Exception as e:
        emit_sse(session_id, "error", {"message": str(e)})
        try:
            recruitment_graph.update_state(
                config,
                {"status": "error", "messages": [HumanMessage(content=f"错误: {str(e)}")]}
            )
        except Exception:
            pass
    finally:
        with _lock:
            _running_sessions.discard(session_id)


# ============================================================
# SSE 端点
# ============================================================
@router.get("/{session_id}/stream")
async def stream_events(session_id: str):
    """GET /api/recruitment/{session_id}/stream — SSE 实时事件流。"""
    q: queue.Queue = queue.Queue(maxsize=100)
    _sse_subscribers[session_id].append(q)

    async def event_generator():
        try:
            while True:
                try:
                    event = q.get_nowait()
                    event_type = event["event"]
                    data = json.dumps(event["data"], ensure_ascii=False)
                    yield f"event: {event_type}\ndata: {data}\n\n"
                    if event_type in ("finished", "error"):
                        break
                except queue.Empty:
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(0.8)
        finally:
            if q in _sse_subscribers.get(session_id, []):
                _sse_subscribers[session_id].remove(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 启动流程
# ============================================================
@router.post("/start", response_model=StartResponse)
def start_recruitment(req: StartRequest, background_tasks: BackgroundTasks):
    """POST /api/recruitment/start — 提交 JD + 简历，创建会话并启动。"""
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "session_id": session_id,
        "jd_text": req.jd_text,
        "resume_text": req.resume_text,
        # 新增字段初始化
        "parsed_jd": {},
        "parsed_resume": {},
        "score": 0,
        "score_reason": "",
        "evidence": [],
        "screen_route": "",
        "interview_questions": [],
        "interview_answers": [],
        "interviewer_feedbacks": [],
        "merged_feedback": {},
        "evaluation": {},
        "recommendation": "",
        "recommendation_reason": "",
        "offer_package": {},
        "final_report": "",
        "status": "running",
        "current_step": "",
        "hr_decisions": {},
        "messages": [HumanMessage(content="招聘流程已启动")],
    }

    with _lock:
        _running_sessions.add(session_id)

    background_tasks.add_task(_run_graph_streamed, session_id, initial_state, config)
    return StartResponse(session_id=session_id, status="running")


# ============================================================
# 查询状态
# ============================================================
@router.get("/{session_id}/state", response_model=StateResponse)
def get_recruitment_state(session_id: str):
    """GET /api/recruitment/{session_id}/state — 查询当前状态。"""
    config = {"configurable": {"thread_id": session_id}}

    try:
        snapshot = recruitment_graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="会话不存在")

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="会话不存在")

    values = snapshot.values
    next_nodes = snapshot.next

    paused_at = None
    status = values.get("status", "running")

    with _lock:
        is_running = session_id in _running_sessions

    if is_running:
        status = "running"
    elif next_nodes and status not in ("completed", "rejected", "error"):
        paused_at = next_nodes[0] if next_nodes else None
        status = "paused"

    return StateResponse(
        session_id=session_id,
        status=status,
        paused_at=paused_at,
        current_step=values.get("current_step", ""),
        parsed_jd=values.get("parsed_jd"),
        parsed_resume=values.get("parsed_resume"),
        score=values.get("score"),
        score_reason=values.get("score_reason"),
        evidence=values.get("evidence"),
        screen_route=values.get("screen_route"),
        interview_questions=values.get("interview_questions"),
        interviewer_feedbacks=values.get("interviewer_feedbacks"),
        merged_feedback=values.get("merged_feedback"),
        evaluation=values.get("evaluation"),
        recommendation=values.get("recommendation"),
        recommendation_reason=values.get("recommendation_reason"),
        offer_package=values.get("offer_package"),
        hr_action_required=HR_ACTION_MAP.get(paused_at) if paused_at else None,
        hr_decisions=values.get("hr_decisions"),
    )


# ============================================================
# 录入面试回答
# ============================================================
@router.post("/{session_id}/answers")
def submit_answers(session_id: str, req: AnswersRequest):
    """POST /api/recruitment/{session_id}/answers — 录入候选人面试回答。"""
    config = {"configurable": {"thread_id": session_id}}

    try:
        snapshot = recruitment_graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="会话不存在")

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="会话不存在")

    if not req.answers:
        raise HTTPException(status_code=422, detail="回答列表不能为空")

    recruitment_graph.update_state(config, {"interview_answers": req.answers})
    return {"message": "面试回答已录入", "answer_count": len(req.answers)}


# ============================================================
# 追加面试题
# ============================================================
@router.post("/{session_id}/questions")
def add_questions(session_id: str, req: QuestionsRequest):
    """POST /api/recruitment/{session_id}/questions — 追加自定义面试题。"""
    config = {"configurable": {"thread_id": session_id}}

    try:
        snapshot = recruitment_graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="会话不存在")

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="会话不存在")

    next_nodes = snapshot.next or []
    if not next_nodes or next_nodes[0] != "interview_gate":
        raise HTTPException(status_code=409, detail="当前不在面试审核阶段，无法追加面试题")

    if not req.questions:
        raise HTTPException(status_code=422, detail="问题列表不能为空")

    existing = list(snapshot.values.get("interview_questions", []))
    # 转为新格式 dict
    for q_text in req.questions:
        existing.append({
            "question": q_text,
            "intent": "HR 自定义",
            "rubric": {"excellent": "优秀", "good": "良好", "poor": "需改进"},
            "_custom": True,
        })

    recruitment_graph.update_state(config, {"interview_questions": existing})

    emit_sse(session_id, "questions_updated", {
        "interview_questions": existing,
        "added_count": len(req.questions),
    })

    return {"message": f"已追加 {len(req.questions)} 道面试题", "total": len(existing)}


# ============================================================
# 提交面试官反馈（PRD: collect_feedback）
# ============================================================
@router.post("/{session_id}/feedback")
def submit_feedback(session_id: str, req: FeedbackRequest):
    """POST /api/recruitment/{session_id}/feedback — 面试官提交反馈。"""
    config = {"configurable": {"thread_id": session_id}}

    try:
        snapshot = recruitment_graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="会话不存在")

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 可以在 collect_feedback 暂停时提交
    next_nodes = snapshot.next or []
    if not next_nodes or next_nodes[0] != "collect_feedback":
        raise HTTPException(status_code=409, detail="当前不在面试官反馈收集阶段")

    existing_feedbacks = list(snapshot.values.get("interviewer_feedbacks", []))
    existing_feedbacks.append({
        "interviewer": req.interviewer,
        "scores": req.scores,
        "overall": req.overall,
    })

    recruitment_graph.update_state(config, {"interviewer_feedbacks": existing_feedbacks})

    emit_sse(session_id, "feedback_submitted", {
        "interviewer": req.interviewer,
        "feedback_count": len(existing_feedbacks),
    })

    return {
        "message": f"面试官 {req.interviewer} 的反馈已录入",
        "total_feedbacks": len(existing_feedbacks),
    }


# ============================================================
# HR 审核决定 + 恢复流程
# ============================================================
@router.post("/{session_id}/resume")
def resume_recruitment(session_id: str, req: ResumeRequest, background_tasks: BackgroundTasks):
    """POST /api/recruitment/{session_id}/resume — HR 审核后恢复流程。"""
    config = {"configurable": {"thread_id": session_id}}

    try:
        snapshot = recruitment_graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="会话不存在")

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="会话不存在")

    values = snapshot.values
    next_nodes = snapshot.next

    status = values.get("status", "")
    if status in ("completed", "rejected"):
        raise HTTPException(status_code=409, detail=f"流程已结束（{status}），无法继续")

    if not next_nodes:
        raise HTTPException(status_code=409, detail="当前没有待执行的节点")

    paused_at = next_nodes[0] if next_nodes else ""

    # HITL #2 校验：approved 时必须有回答
    if paused_at == "interview_gate" and req.decision == "approved":
        answers = values.get("interview_answers", [])
        if not answers:
            raise HTTPException(status_code=422, detail="请先录入候选人回答")

    # 映射暂停点到决策 key
    decision_key_map = {
        "screening_gate": "screening",
        "interview_gate": "interview",
        "collect_feedback": "feedback",
        "decision_gate": "final",
    }
    decision_key = decision_key_map.get(paused_at, "unknown")

    hr_decisions = dict(values.get("hr_decisions", {}))
    hr_decisions[decision_key] = req.decision

    # 如果 HR 在 screening_gate 指定了分流方向
    update = {"hr_decisions": hr_decisions, "status": "running"}
    if req.screen_route and paused_at == "screening_gate":
        hr_decisions["screen_route"] = req.screen_route

    if req.notes:
        update["messages"] = [HumanMessage(content=f"[HR 备注] {req.notes}")]

    recruitment_graph.update_state(config, update)

    with _lock:
        _running_sessions.add(session_id)

    emit_sse(session_id, "status", {
        "status": "running",
        "message": f"HR 已决定: {req.decision}，流程恢复中...",
    })

    background_tasks.add_task(_run_graph_streamed, session_id, None, config)
    return {"message": f"HR 决定: {req.decision}，流程恢复中", "decision": req.decision}


# ============================================================
# 获取最终报告
# ============================================================
@router.get("/{session_id}/report", response_model=ReportResponse)
def get_report(session_id: str):
    """GET /api/recruitment/{session_id}/report — 获取最终报告。"""
    config = {"configurable": {"thread_id": session_id}}

    try:
        snapshot = recruitment_graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="会话不存在")

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="会话不存在")

    values = snapshot.values
    if values.get("status") not in ("completed", "rejected"):
        raise HTTPException(status_code=409, detail="流程尚未完成，暂无报告")

    return ReportResponse(
        session_id=session_id,
        final_report=values.get("final_report", ""),
        recommendation=values.get("recommendation", ""),
        hr_decisions=values.get("hr_decisions", {}),
        offer_package=values.get("offer_package"),
    )
