"""
招聘助手 - 智能体节点（PRD 对齐版）
======================================
包含 12 个节点函数，对齐 PRD 8 节点架构：
  6 个 LLM 智能体：parse_jd, resume_parser, screener, interview_generator,
                    merge_feedback, offer_pack
  3 个 gate 门节点：screening_gate, interview_gate, decision_gate（不调用 LLM）
  1 个 feedback 门：collect_feedback（HITL 多面试官反馈收集）
  1 个评估节点：evaluator（LLM 评估面试回答）
  1 个报告生成：report_generator（模板拼接）
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .state import RecruitmentState
from .tools import get_llm, invoke_for_json


# ============================================================
# LLM 实例（模块级，所有节点共用）
# ============================================================
llm = get_llm(temperature=0.3, json_mode=True)


# ============================================================
# 节点 1: parse_jd — 解析/规范化岗位画像（PRD: parse_jd）
# ============================================================
def parse_jd(state: RecruitmentState) -> dict:
    """
    从 JD 原文中提取结构化岗位要求。
    输出：{title, required_skills, nice_to_have_skills,
           experience_years, education, responsibilities}
    """
    parsed = invoke_for_json(llm, [
        SystemMessage(content=(
            "You are a job description parsing specialist. Extract structured requirements from the JD text. "
            "Return ONLY valid JSON with these fields:\n"
            '{"title": "job title string", '
            '"required_skills": ["must-have skill 1", "must-have skill 2"], '
            '"nice_to_have_skills": ["bonus skill 1", "bonus skill 2"], '
            '"experience_years": "e.g. 3-5年 or 5年以上", '
            '"education": "e.g. 本科及以上", '
            '"responsibilities": ["responsibility 1", "responsibility 2"]}'
        )),
        HumanMessage(content=f"Parse this Job Description:\n\n{state['jd_text']}"),
    ])

    return {
        "parsed_jd": parsed,
        "current_step": "parse_jd",
        "status": "running",
        "messages": [AIMessage(content=f"[JD解析] 已完成岗位画像结构化提取")],
    }


# ============================================================
# 节点 2: resume_parser — 解析简历
# ============================================================
def resume_parser(state: RecruitmentState) -> dict:
    """
    从简历原文中提取结构化信息。
    输出 JSON：{name, phone, email, skills, experience, education, summary}
    """
    parsed = invoke_for_json(llm, [
        SystemMessage(content=(
            "You are a resume parsing specialist. Extract structured information from the resume text. "
            "Return ONLY valid JSON with these fields:\n"
            '{"name": "string", "phone": "string or null", "email": "string or null", '
            '"skills": ["skill1", "skill2"], '
            '"experience": [{"company": "string", "role": "string", "duration": "string", "description": "string"}], '
            '"education": [{"school": "string", "degree": "string", "major": "string"}], '
            '"summary": "one sentence summary of the candidate"}'
        )),
        HumanMessage(content=f"Parse this resume:\n\n{state['resume_text']}"),
    ])

    return {
        "parsed_resume": parsed,
        "current_step": "resume_parser",
        "status": "running",
        "messages": [AIMessage(content="[简历解析] 已完成简历结构化提取")],
    }


# ============================================================
# 节点 3: screener — 筛选评分（PRD: screen_resume，带证据引用）
# ============================================================
def screener(state: RecruitmentState) -> dict:
    """
    对比结构化 JD 和简历，逐项评估匹配度，输出分数和证据引用。
    新增：evidence 字段提供逐项对照。
    新增：screen_route 提供 4 路分流建议。
    """
    parsed_jd = state.get("parsed_jd", {})
    parsed_resume = state.get("parsed_resume", {})

    result = invoke_for_json(llm, [
        SystemMessage(content=(
            "You are a recruitment screening specialist. Compare the structured job requirements "
            "with the candidate's resume. For EACH requirement, provide evidence from the resume.\n\n"
            "Return ONLY valid JSON with this structure:\n"
            '{\n'
            '  "score": <integer 0-100>,\n'
            '  "reason": "总结性评价（中文）",\n'
            '  "evidence": [\n'
            '    {"requirement": "要求描述", "matched": true/false, '
            '"citation": "从简历中引用的证据原文", "comment": "匹配分析说明（中文）"}\n'
            '  ],\n'
            '  "screen_route": "reject/phone_screen/onsite/need_more_info"\n'
            '}\n\n'
            "screen_route rules:\n"
            "- score >= 80: 'onsite' (直接现场面试)\n"
            "- score >= 50: 'phone_screen' (电话面试)\n"
            "- score < 30: 'reject' (淘汰)\n"
            "- otherwise or if key info missing: 'need_more_info' (需要补充信息)"
        )),
        HumanMessage(content=(
            f"Job Requirements (structured):\n{json.dumps(parsed_jd, ensure_ascii=False, indent=2)}\n\n"
            f"Candidate Resume (structured):\n{json.dumps(parsed_resume, ensure_ascii=False, indent=2)}"
        )),
    ])
    score = result.get("score", 0) if isinstance(result, dict) else 0
    reason = result.get("reason", str(result)) if isinstance(result, dict) else str(result)
    evidence = result.get("evidence", []) if isinstance(result, dict) else []
    screen_route = result.get("screen_route", "phone_screen") if isinstance(result, dict) else "phone_screen"

    # 校验 screen_route 合法性
    valid_routes = {"reject", "phone_screen", "onsite", "need_more_info"}
    if screen_route not in valid_routes:
        screen_route = "phone_screen"

    return {
        "score": score,
        "score_reason": reason,
        "evidence": evidence,
        "screen_route": screen_route,
        "current_step": "screener",
        "messages": [AIMessage(content=f"[筛选评分] 匹配度: {score}/100，建议分流: {screen_route}")],
    }


# ============================================================
# 节点 4: screening_gate — 筛选审核门（PRD: route_screen）
# ============================================================
def screening_gate(state: RecruitmentState) -> dict:
    """
    Pass-through 门节点。不调用 LLM。
    HR 在此审核筛选结果，并可修改 screen_route 分流决定。
    条件边在此节点之后触发 4 路路由。
    """
    return {"current_step": "screening_gate"}


# ============================================================
# 节点 5: interview_generator — 生成面试题+评分标准
# （PRD: generate_questions，含 rubrics）
# ============================================================
def interview_generator(state: RecruitmentState) -> dict:
    """
    根据简历、JD 和分流类型生成面试题及评分标准（rubrics）。
    screen_route 为 phone_screen 或 onsite 决定题目风格。
    """
    route = state.get("screen_route", "phone_screen")
    question_type = "电话面试" if route == "phone_screen" else "现场面试"
    num_questions = "5-6" if route == "phone_screen" else "6-8"

    parsed_jd = state.get("parsed_jd", {})
    parsed_resume = state.get("parsed_resume", {})

    questions = invoke_for_json(llm, [
        SystemMessage(content=(
            f"You are an interview question specialist. Generate {num_questions} targeted "
            f"{question_type} questions based on the job requirements and candidate profile.\n\n"
            "For each question, provide:\n"
            "- The question text\n"
            "- The intent (what it tests)\n"
            "- A rubric with 3 levels: excellent, good, poor\n\n"
            "Return ONLY valid JSON array:\n"
            '[{"question": "问题文本（中文）", "intent": "考察意图（中文）", '
            '"rubric": {"excellent": "优秀回答标准", "good": "良好回答标准", "poor": "不佳回答标准"}}]'
        )),
        HumanMessage(content=(
            f"Job Requirements:\n{json.dumps(parsed_jd, ensure_ascii=False, indent=2)}\n\n"
            f"Candidate:\n{json.dumps(parsed_resume, ensure_ascii=False, indent=2)}\n\n"
            f"Screening Score: {state.get('score', 0)}/100\n"
            f"Interview Type: {question_type}"
        )),
    ])
    if not isinstance(questions, list):
        questions = [{"question": str(questions), "intent": "综合评估",
                      "rubric": {"excellent": "优秀", "good": "良好", "poor": "需改进"}}]

    # 确保每个 question 都有完整结构
    normalized = []
    for q in questions:
        if isinstance(q, str):
            q = {"question": q, "intent": "综合评估",
                 "rubric": {"excellent": "优秀", "good": "良好", "poor": "需改进"}}
        elif isinstance(q, dict) and "question" not in q:
            continue
        normalized.append(q)

    return {
        "interview_questions": normalized,
        "current_step": "interview_generator",
        "messages": [AIMessage(content=f"[面试题生成] 已生成 {len(normalized)} 道{question_type}题（含评分标准）")],
    }


# ============================================================
# 节点 6: interview_gate — 面试审核门（pass-through）
# ============================================================
def interview_gate(state: RecruitmentState) -> dict:
    """
    Pass-through 门节点。不调用 LLM。
    在此节点之前 HITL 暂停，HR 审核面试题并录入候选人回答。
    """
    return {"current_step": "interview_gate"}


# ============================================================
# 节点 7: evaluator — 面试评估
# ============================================================
def evaluator(state: RecruitmentState) -> dict:
    """
    逐题评估候选人的面试回答，结合 rubrics 进行打分。
    """
    questions = state.get("interview_questions", [])
    answers = state.get("interview_answers", [])

    # 构建 Q&A 对（兼容新旧格式）
    qa_pairs = []
    for i, q in enumerate(questions):
        q_text = q["question"] if isinstance(q, dict) else str(q)
        rubric = q.get("rubric", {}) if isinstance(q, dict) else {}
        a = answers[i] if i < len(answers) else "(未回答)"
        qa_pairs.append({
            "question": q_text,
            "answer": a,
            "rubric": rubric,
        })

    evaluation = invoke_for_json(llm, [
        SystemMessage(content=(
            "You are an interview evaluator. For each question-answer pair, "
            "evaluate against the provided rubric. Provide a score (1-10) and comment.\n"
            "Also give overall_score (1-10) and overall_comment.\n"
            "Return ONLY valid JSON:\n"
            '{"items": [{"question": "...", "answer": "...", "score": <1-10>, "comment": "评价（中文）"}], '
            '"overall_score": <1-10>, "overall_comment": "综合评价（中文）"}'
        )),
        HumanMessage(content=(
            f"Job Description:\n{state['jd_text']}\n\n"
            f"Q&A Pairs (with rubrics):\n{json.dumps(qa_pairs, ensure_ascii=False, indent=2)}"
        )),
    ])
    if not isinstance(evaluation, dict):
        evaluation = {"_raw": str(evaluation)}

    return {
        "evaluation": evaluation,
        "current_step": "evaluator",
        "messages": [AIMessage(content="[面试评估] 评估完成")],
    }


# ============================================================
# 节点 8: collect_feedback — 多面试官反馈收集门（PRD: collect_feedback）
# ============================================================
def collect_feedback(state: RecruitmentState) -> dict:
    """
    Pass-through HITL 门节点。不调用 LLM。
    在此节点之前暂停，等待多位面试官通过 API 提交反馈。
    面试官反馈通过 update_state 写入 interviewer_feedbacks 字段。
    """
    return {"current_step": "collect_feedback"}


# ============================================================
# 节点 9: merge_feedback — 合并多面试官反馈（PRD: merge_feedback）
# ============================================================
def merge_feedback(state: RecruitmentState) -> dict:
    """
    合并多位面试官的反馈，分析共识与分歧，给出补面建议。
    如果只有单面试官或使用旧格式 evaluation，直接转换。
    """
    feedbacks = state.get("interviewer_feedbacks", [])
    evaluation = state.get("evaluation", {})

    # 如果没有多面试官反馈，从 evaluation 构建兼容数据
    if not feedbacks and evaluation and "items" in evaluation:
        merged = {
            "consensus_score": evaluation.get("overall_score", 0),
            "disagreements": [],
            "strengths": [],
            "concerns": [],
            "follow_up_suggestions": [],
            "summary": evaluation.get("overall_comment", ""),
        }
        return {
            "merged_feedback": merged,
            "current_step": "merge_feedback",
            "messages": [AIMessage(content="[面评合并] 单面试官评估，无需合并分歧")],
        }

    # 多面试官反馈 → LLM 合并分析
    merged = invoke_for_json(llm, [
        SystemMessage(content=(
            "You are a recruitment feedback analyst. Analyze feedback from multiple interviewers.\n"
            "Identify consensus, disagreements, strengths, concerns, and suggest follow-up actions.\n"
            "Return ONLY valid JSON:\n"
            '{\n'
            '  "consensus_score": <float, average score>,\n'
            '  "disagreements": [{"question_idx": <int>, "scores": [<int>], '
            '"analysis": "分歧原因分析（中文）"}],\n'
            '  "strengths": ["优势1（中文）", "优势2"],\n'
            '  "concerns": ["顾虑1（中文）", "顾虑2"],\n'
            '  "follow_up_suggestions": ["补面建议1（中文）"],\n'
            '  "summary": "综合分析总结（中文）"\n'
            '}'
        )),
        HumanMessage(content=(
            f"Interview Questions:\n{json.dumps(state.get('interview_questions', []), ensure_ascii=False, indent=2)}\n\n"
            f"Interviewer Feedbacks:\n{json.dumps(feedbacks, ensure_ascii=False, indent=2)}"
        )),
    ])
    if not isinstance(merged, dict):
        merged = {"consensus_score": 0, "disagreements": [], "strengths": [],
                  "concerns": [], "follow_up_suggestions": [], "summary": str(merged)}

    return {
        "merged_feedback": merged,
        "current_step": "merge_feedback",
        "messages": [AIMessage(
            content=f"[面评合并] 已合并 {len(feedbacks)} 位面试官反馈，"
                    f"发现 {len(merged.get('disagreements', []))} 处分歧"
        )],
    }


# ============================================================
# 节点 10: advisor — 录用建议（PRD: decision_gate 的 LLM 部分）
# ============================================================
def advisor(state: RecruitmentState) -> dict:
    """
    综合所有信息给出录用建议。
    使用 merged_feedback（如有）或 evaluation 进行综合评判。
    """
    # 优先使用合并后的面评
    feedback_data = state.get("merged_feedback", {})
    if not feedback_data:
        feedback_data = state.get("evaluation", {})

    result = invoke_for_json(llm, [
        SystemMessage(content=(
            "You are a senior recruitment advisor. Based on all available information, provide a hiring recommendation.\n"
            "Choose one of: 强烈推荐 / 建议录用 / 谨慎考虑 / 不建议录用.\n"
            "Return ONLY valid JSON:\n"
            '{"recommendation": "one of the four options", "reason": "详细理由（中文），包含风险点分析"}'
        )),
        HumanMessage(content=(
            f"Job Requirements:\n{json.dumps(state.get('parsed_jd', {}), ensure_ascii=False, indent=2)}\n\n"
            f"Candidate:\n{json.dumps(state.get('parsed_resume', {}), ensure_ascii=False, indent=2)}\n\n"
            f"Screening: {state.get('score', 0)}/100 - {state.get('score_reason', 'N/A')}\n\n"
            f"Evidence:\n{json.dumps(state.get('evidence', []), ensure_ascii=False, indent=2)}\n\n"
            f"Interview Feedback:\n{json.dumps(feedback_data, ensure_ascii=False, indent=2)}"
        )),
    ])
    recommendation = result.get("recommendation", "谨慎考虑") if isinstance(result, dict) else "谨慎考虑"
    reason = result.get("reason", str(result)) if isinstance(result, dict) else str(result)

    return {
        "recommendation": recommendation,
        "recommendation_reason": reason,
        "current_step": "advisor",
        "messages": [AIMessage(content=f"[录用建议] {recommendation}")],
    }


# ============================================================
# 节点 11: decision_gate — 人工决策门（PRD: decision_gate）
# ============================================================
def decision_gate(state: RecruitmentState) -> dict:
    """
    Pass-through HITL 门节点。不调用 LLM。
    HR 在此做最终录用决策：offer / rejected / continue（补面）。
    条件边在此节点之后路由到 offer_pack 或 report_generator。
    """
    return {"current_step": "decision_gate"}


# ============================================================
# 节点 12: offer_pack — 生成 Offer 信息包（PRD: offer_pack）
# ============================================================
def offer_pack(state: RecruitmentState) -> dict:
    """
    仅在 HR 决定 offer 时到达。
    生成 offer 话术、薪资建议、入职流程清单。
    """
    parsed_jd = state.get("parsed_jd", {})
    parsed_resume = state.get("parsed_resume", {})

    package = invoke_for_json(llm, [
        SystemMessage(content=(
            "You are a recruitment offer specialist. Generate an offer package including:\n"
            "1. Offer talking points (how to present the offer to the candidate)\n"
            "2. Salary suggestion (based on role and candidate experience)\n"
            "3. Onboarding checklist (steps for new hire)\n"
            "4. Suggested start date\n\n"
            "Return ONLY valid JSON:\n"
            '{\n'
            '  "offer_talking_points": ["话术要点1（中文）", "话术要点2"],\n'
            '  "salary_suggestion": "薪资建议说明（中文）",\n'
            '  "onboarding_checklist": ["入职步骤1（中文）", "入职步骤2"],\n'
            '  "start_date_suggestion": "建议入职日期说明（中文）"\n'
            '}'
        )),
        HumanMessage(content=(
            f"Job Title: {parsed_jd.get('title', 'N/A')}\n"
            f"Experience Required: {parsed_jd.get('experience_years', 'N/A')}\n\n"
            f"Candidate: {parsed_resume.get('name', 'N/A')}\n"
            f"Skills: {', '.join(parsed_resume.get('skills', []))}\n"
            f"Experience: {json.dumps(parsed_resume.get('experience', []), ensure_ascii=False)}\n\n"
            f"Screening Score: {state.get('score', 0)}/100\n"
            f"Recommendation: {state.get('recommendation', 'N/A')}"
        )),
    ])
    if not isinstance(package, dict):
        package = {"offer_talking_points": [], "salary_suggestion": str(package),
                   "onboarding_checklist": [], "start_date_suggestion": "待定"}

    return {
        "offer_package": package,
        "current_step": "offer_pack",
        "messages": [AIMessage(content="[Offer生成] 已生成 Offer 信息包")],
    }


# ============================================================
# 节点 13: report_generator — 生成最终报告
# ============================================================
def report_generator(state: RecruitmentState) -> dict:
    """
    汇总所有字段生成 Markdown 格式的完整报告。
    不调用 LLM，纯模板拼接。
    """
    hr = state.get("hr_decisions", {})
    parsed_jd = state.get("parsed_jd", {})
    parsed_resume = state.get("parsed_resume", {})

    is_rejected = any(v == "rejected" for v in hr.values())
    is_offer = hr.get("final") == "offer"
    final_status = "rejected" if is_rejected else "completed"

    report_lines = [
        "# 招聘评估报告",
        "",
        f"## 岗位信息",
        f"- **职位**：{parsed_jd.get('title', 'N/A')}",
        f"- **必备技能**：{', '.join(parsed_jd.get('required_skills', []))}",
        f"- **经验要求**：{parsed_jd.get('experience_years', 'N/A')}",
        "",
        "## 候选人信息",
        f"- **姓名**：{parsed_resume.get('name', '未知')}",
        f"- **技能**：{', '.join(parsed_resume.get('skills', []))}",
        f"- **摘要**：{parsed_resume.get('summary', 'N/A')}",
        "",
        "## 筛选评分",
        f"- **匹配度**：{state.get('score', 'N/A')}/100",
        f"- **分流建议**：{state.get('screen_route', 'N/A')}",
        f"- **评分理由**：{state.get('score_reason', 'N/A')}",
        f"- **HR 决定**：{hr.get('screening', 'N/A')}",
        "",
    ]

    # 证据引用
    evidence = state.get("evidence", [])
    if evidence:
        report_lines.append("### 匹配证据")
        for e in evidence:
            if isinstance(e, dict):
                status = "✅" if e.get("matched") else "❌"
                report_lines.append(f"- {status} **{e.get('requirement', '')}**：{e.get('citation', '')} — {e.get('comment', '')}")
        report_lines.append("")

    # 面试环节
    questions = state.get("interview_questions", [])
    if questions:
        report_lines.extend([
            "## 面试环节",
            f"- **面试题数量**：{len(questions)}",
            f"- **面试类型**：{'电话面试' if state.get('screen_route') == 'phone_screen' else '现场面试'}",
            f"- **HR 决定**：{hr.get('interview', 'N/A')}",
            "",
        ])

    # 面评合并
    merged = state.get("merged_feedback", {})
    if merged and merged.get("summary"):
        report_lines.extend([
            "## 面评分析",
            f"- **共识评分**：{merged.get('consensus_score', 'N/A')}",
            f"- **优势**：{', '.join(merged.get('strengths', []))}",
            f"- **顾虑**：{', '.join(merged.get('concerns', []))}",
            f"- **综合**：{merged.get('summary', '')}",
            "",
        ])
        if merged.get("disagreements"):
            report_lines.append("### 分歧点")
            for d in merged["disagreements"]:
                report_lines.append(f"- 问题{d.get('question_idx', '?')}：{d.get('analysis', '')}")
            report_lines.append("")

    # 面试评估（兼容旧版）
    evaluation = state.get("evaluation", {})
    if evaluation and "overall_score" in evaluation and not merged:
        report_lines.extend([
            "## 面试评估",
            f"- **综合评分**：{evaluation.get('overall_score', 'N/A')}/10",
            f"- **综合评价**：{evaluation.get('overall_comment', 'N/A')}",
            "",
        ])

    # 录用建议
    if state.get("recommendation"):
        report_lines.extend([
            "## 录用建议",
            f"- **建议**：{state['recommendation']}",
            f"- **理由**：{state.get('recommendation_reason', 'N/A')}",
            f"- **HR 最终决策**：{hr.get('final', 'N/A')}",
            "",
        ])

    # Offer 信息
    offer = state.get("offer_package", {})
    if is_offer and offer:
        report_lines.extend([
            "## Offer 信息",
            f"- **薪资建议**：{offer.get('salary_suggestion', 'N/A')}",
            f"- **建议入职时间**：{offer.get('start_date_suggestion', 'N/A')}",
            "",
        ])
        if offer.get("offer_talking_points"):
            report_lines.append("### Offer 话术要点")
            for point in offer["offer_talking_points"]:
                report_lines.append(f"- {point}")
            report_lines.append("")
        if offer.get("onboarding_checklist"):
            report_lines.append("### 入职流程清单")
            for step in offer["onboarding_checklist"]:
                report_lines.append(f"- [ ] {step}")
            report_lines.append("")

    # 最终结论
    if is_offer:
        conclusion = "✅ 录用（已生成 Offer）"
    elif is_rejected:
        conclusion = "❌ 未通过"
    else:
        conclusion = "✅ 通过"

    report_lines.extend([
        "## 最终结论",
        "",
        f"**{conclusion}**",
        "",
        "---",
        "*报告由招聘助手自动生成*",
    ])

    return {
        "final_report": "\n".join(report_lines),
        "status": final_status,
        "current_step": "done",
        "messages": [AIMessage(content=f"[报告生成] 最终结论: {conclusion}")],
    }
