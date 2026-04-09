## LangGraph 往往更贴合

它本质是“带状态的图（state machine / DAG）+ 节点函数”，你可以把每个招聘环节做成节点，把候选人状态（简历、评分、面试记录、决策）放在统一 state 里流转。

## LangGraph 适合的点

- 流程可控 ：每一步是显式节点，天然支持分支（通过/淘汰/补面/转岗）、循环（补充信息→再评估）。
- 状态管理清晰 ：把候选人评估结果用结构化字段放在 state，便于追踪与回放。
- 便于加人工确认（HITL） ：在“淘汰/发 offer/薪资建议”等节点强制停下来等人批。
- 可测试 ：节点就是函数，更容易做单元测试、回归测试。
-

## 招聘场景下用 LangGraph 的一个参考图（概念）

1. parse_jd ：解析/规范化岗位画像（结构化输出）
2. screen_resume ：简历匹配评分（必须带证据引用）
3. route_screen ：分流（淘汰/电话面/直接 onsite/补充信息）
4. generate_questions ：生成电话面问题包 + rubrics
5. collect_feedback ：输入面试官评分（人填）→写入 state
6. merge_feedback ：合并多轮面评，输出分歧点与补面建议
7. decision_gate ：人工确认（继续/淘汰/offer）
8. offer_pack ：生成 offer 话术、入职流程清单（可选）
