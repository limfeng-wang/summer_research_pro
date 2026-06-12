"""
H-RAMOS ABABC 共享工具模块
=======================
所有四个流水线文件 (03排列组合, 04优化, 05全量标注, 06验证集) 统一从此导入。
消除之前各文件独立copy-paste导致的 _parse_b_verdicts 关键词列表和回退逻辑不一致。

提供的公共接口:
    - get_dim_label / get_dim_confidence / get_dim_reasoning : 维度数据安全提取
    - parse_b_verdicts : B(Reviewer)审核结论统一解析
    - build_dim_trace_empty / detect_changed_dimensions : 维度级轨迹结构
    - build_final_from_dim_trace / extract_trace_meta : 结果构建与元信息提取
    - run_ababc_pipeline : 统一ABABC流水线（推荐所有文件使用此函数）
"""

import json as _json

# ============================================================
# 一、维度标签/置信度/推理 安全提取
#    处理各种异常输入（None、非dict、缺失键），永不抛异常
# ============================================================

def get_dim_label(output: dict, dim_name: str) -> int:
    """从LLM输出中安全提取某维度的二分类标签 (0/1)。
    - output 不是 dict → 返回 0
    - 该维度不存在或不是 dict → 返回 0
    - label 值匹配 "1"/"true"/"yes" → 返回 1，否则 0
    """
    if not isinstance(output, dict):
        return 0
    dim_data = output.get(dim_name, {})
    if not isinstance(dim_data, dict):
        return 0
    val = str(dim_data.get("label", "0")).lower()
    return 1 if val in ["1", "true", "yes"] else 0


def get_dim_confidence(output: dict, dim_name: str) -> float:
    """从LLM输出中安全提取某维度的置信度 (2.0-5.0)。
    解析失败或缺失时返回 0.0。
    """
    if not isinstance(output, dict):
        return 0.0
    dim_data = output.get(dim_name, {})
    if not isinstance(dim_data, dict):
        return 0.0
    try:
        return float(dim_data.get("confidence", 0))
    except (ValueError, TypeError):
        return 0.0


def get_dim_reasoning(output: dict, dim_name: str) -> str:
    """从LLM输出中安全提取某维度的推理文本。
    缺失时返回空字符串。
    """
    if not isinstance(output, dict):
        return ""
    dim_data = output.get(dim_name, {})
    if not isinstance(dim_data, dict):
        return ""
    return str(dim_data.get("reasoning", ""))


# ============================================================
# 二、B (Reviewer) 审核结论解析（统一版本）
#    所有四个文件的唯一 B 判决策略实现
# ============================================================

# 拒绝信号关键词 —— 先检查，优先级高
# 匹配到任意一个即判定为"不通过"
REJECT_KEYWORDS = [
    "incorrect", "wrong", "false", "missed", "error", "mistake", "fail",
    "错误", "不正确", "不对", "有误", "不准确", "错了", "漏判",
    "reject", "disapproved", "not approved", "not correct",
    "hallucination", "missing", "invalid", "not found", "should be",
]

# 通过信号关键词 —— 后检查，仅在无拒绝信号时生效
PASS_KEYWORDS = [
    "correct", "pass", "ok", "yes", "good", "fine", "right",
    "正确", "通过", "无误", "可以", "没问题", "对的",
    "approved", "approve", "accept", "valid", "accurate",
]


def parse_b_verdicts(b_output: dict, dim_names: list) -> dict:
    """统一解析 Reviewer 的维度级审核结论。

    判决策略（按优先级）:
        1. dimension_feedback 字段存在 → 先匹配拒绝词 → 再匹配通过词
        2. 有反馈但无明确信号 → 保守拒绝（宁可多走一轮A2也不错放漏判）
        3. 无 dimension_feedback → 降级使用全局 is_correct 字段
        4. b_output 不是 dict → 全部默认通过（防御性兜底）

    Args:
        b_output: Reviewer 的原始 JSON 输出
        dim_names: 维度名列表

    Returns:
        {dim_name: {"approved": bool, "feedback": str}}
    """
    verdicts = {}
    if not isinstance(b_output, dict):
        return {d: {"approved": True, "feedback": ""} for d in dim_names}

    dim_feedback = b_output.get("dimension_feedback", {})
    if isinstance(dim_feedback, dict) and len(dim_feedback) > 0:
        for dim_name in dim_names:
            fb = str(dim_feedback.get(dim_name, ""))
            if not fb:
                # 该维度无反馈 → 视为通过
                verdicts[dim_name] = {"approved": True, "feedback": ""}
            elif any(kw in fb.lower() for kw in REJECT_KEYWORDS):
                verdicts[dim_name] = {"approved": False, "feedback": fb}
            elif any(kw in fb.lower() for kw in PASS_KEYWORDS):
                verdicts[dim_name] = {"approved": True, "feedback": fb}
            else:
                # 有反馈但无明确信号 → 保守拒绝（宁可多走一轮A2也不错放）
                verdicts[dim_name] = {"approved": False, "feedback": fb}
    else:
        # 降级路径：无 dimension_feedback → 使用全局 is_correct 字段
        global_ok = str(b_output.get("is_correct", "")).lower() == "true"
        for dim_name in dim_names:
            verdicts[dim_name] = {"approved": global_ok, "feedback": ""}
    return verdicts


# ============================================================
# 三、维度级 ABABC 轨迹结构
#    每个维度在每个阶段（A1/B1/A2/B2/C）的状态都被记录
# ============================================================

def build_dim_trace_empty(dim_names: list) -> dict:
    """为每个维度初始化空的轨迹槽位。
    结构: { dim_name: { A1, B1_approved, B1_feedback, A2, B2_approved, B2_feedback, C, final_label, final_source, final_confidence } }
    """
    return {
        d: {
            "A1": None, "B1_approved": None, "B1_feedback": "",
            "A2": None, "B2_approved": None, "B2_feedback": "",
            "C": None,
            "final_label": None, "final_source": None, "final_confidence": None
        }
        for d in dim_names
    }


def detect_changed_dimensions(output_a: dict, output_b: dict, dim_names: list) -> tuple:
    """比较两次标注输出，检测哪些维度的标签发生了变化。

    Args:
        output_a: 前一次标注结果 (如 A1)
        output_b: 后一次标注结果 (如 A2)

    Returns:
        (changed_set: set, changes_dict: {dim_name: {"from": int, "to": int}})
    """
    changed = set()
    changes = {}
    for dim_name in dim_names:
        lab_a = get_dim_label(output_a, dim_name)
        lab_b = get_dim_label(output_b, dim_name)
        if lab_a != lab_b:
            changed.add(dim_name)
            changes[dim_name] = {"from": lab_a, "to": lab_b}
    return changed, changes


def build_final_from_dim_trace(dim_trace: dict, dim_names: list) -> dict:
    """根据每个维度的 final_source 构建最终标注输出。
    - final_source == "A1" → 取 A1 阶段输出
    - final_source == "A2" → 取 A2 阶段输出
    - 其他（C等）→ 取 C，回退到 A2，再回退到 A1
    """
    final = {}
    for dim_name in dim_names:
        dt = dim_trace.get(dim_name, {})
        source = dt.get("final_source", "A1")
        if source == "A1":
            final[dim_name] = dt.get("A1", {})
        elif source == "A2":
            final[dim_name] = dt.get("A2", {})
        else:
            final[dim_name] = dt.get("C", dt.get("A2", dt.get("A1", {})))
    return final


def extract_trace_meta(dim_trace: dict, dim_names: list) -> dict:
    """从维迹中提取元信息，附加到最终输出。
    包括: _final_source (每个维度的最终来源), _b1_approved (B1 审核通过状态)
    供 05_全量标注 在 Excel 中记录溯源信息。
    """
    final_source = {}
    b1_approved = {}
    for d in dim_names:
        dt = dim_trace.get(d, {})
        final_source[d] = dt.get("final_source", "")
        b1_approved[d] = str(dt.get("B1_approved", ""))
    return {"_final_source": final_source, "_b1_approved": b1_approved}


# ============================================================
# 四、统一 ABABC 流水线（四个文件的唯一实现）
#    推荐所有新文件通过此函数执行标注，不再各自复制 ABABC 逻辑
# ============================================================

async def run_ababc_pipeline(
    text: str,
    dim_names: list,
    prompts: dict,              # {"Annotator": str, "Reviewer": str, "Arbitrator": str}
    call_annotator,             # async fn(system_prompt, user_message) -> dict
    call_reviewer,              # async fn(system_prompt, user_message) -> dict
    call_arbitrator,            # async fn(system_prompt, user_message) -> dict
    annotator_user_msg: str = None,   # 自定义 A1 用户消息（默认 f"Text: {text}"）
    extra_context: dict = None,       # 额外上下文注入到 trace（如 country 等）
) -> dict:
    """统一的维度级 ABABC 流程: A1→B1→A2→B2→C。

    流程说明:
        A1: Annotator 首次标注全部维度
        B1: Reviewer 逐维度审核，通过的锁定为 A1，拒绝的进入 A2
        A2: Annotator 仅修订被 B1 拒绝的维度，通过的维度保持 A1 不变
        B2: Reviewer 仅重新审核 A2 实际改变了的维度
        C:  Arbitrator 仅仲裁 B2 仍然拒绝的维度 + A2 未实际改变的维度

    Args:
        text: 待标注的帖子原文
        dim_names: CSM 维度名列表（如 ["Perceived Cause", ...]）
        prompts: 三个角色的 system prompt
        call_annotator/reviewer/arbitrator: 异步调用函数，签名 async fn(system, user) -> dict
        annotator_user_msg: 自定义 A1 阶段用户消息，None 时默认 "Text: {text}"
        extra_context: 额外键值对注入到 trace 根级

    Returns:
        trace dict:
            - Steps: {"A1": ..., "B1": ..., "A2": ..., "B2": ..., "C": ...}
            - DimTrace: {dim_name: {A1, B1_approved, B1_feedback, A2, B2_approved, B2_feedback, C, final_*, ...}}
            - DimChanges: {"A1_to_A2": {dim: {"from": int, "to": int}}}
            - Final_Output: 最终标注结果 (dict, key=dim_name)
            - Exit_Stage: "Consensus_R1" | "Consensus_R2" | "Arbitrated" | "Error_*"
    """
    trace = {
        "Steps": {},
        "DimTrace": build_dim_trace_empty(dim_names),
        "DimChanges": {},
        "Final_Output": {},
        "Exit_Stage": "Unknown"
    }
    if extra_context:
        trace.update(extra_context)

    sys_a = prompts["Annotator"]
    sys_b = prompts["Reviewer"]
    sys_c = prompts["Arbitrator"]

    # ======== A1: 首次标注全部维度 ========
    a1_msg = annotator_user_msg if annotator_user_msg else f"Text: {text}"
    res_a1 = await call_annotator(sys_a, a1_msg)
    trace["Steps"]["A1"] = res_a1
    if not isinstance(res_a1, dict) or "error" in res_a1:
        trace["Exit_Stage"] = "Error_A1"
        return trace

    # 将 A1 结果写入每个维度的轨迹
    for d in dim_names:
        trace["DimTrace"][d]["A1"] = res_a1.get(d, {})
        trace["DimTrace"][d]["final_label"] = get_dim_label(res_a1, d)
        trace["DimTrace"][d]["final_source"] = "A1"
        trace["DimTrace"][d]["final_confidence"] = get_dim_confidence(res_a1, d)

    # ======== B1: 维度级审核（逐一检查 A1 的每个维度）========
    b1_input = _json.dumps({"text": text, "annotation": res_a1}, ensure_ascii=False)
    res_b1 = await call_reviewer(sys_b, b1_input)
    trace["Steps"]["B1"] = res_b1

    if not isinstance(res_b1, dict) or "error" in res_b1:
        # B1 失败：保守策略，直接接受 A1 全部结果
        trace["Final_Output"] = build_final_from_dim_trace(trace["DimTrace"], dim_names)
        trace["Exit_Stage"] = "B1_Error_Fallback"
        return trace

    b1_verdicts = parse_b_verdicts(res_b1, dim_names)
    for d in dim_names:
        v = b1_verdicts.get(d, {"approved": True, "feedback": ""})
        trace["DimTrace"][d]["B1_approved"] = v["approved"]
        trace["DimTrace"][d]["B1_feedback"] = v["feedback"]

    b1_rejected = [d for d in dim_names if not trace["DimTrace"][d]["B1_approved"]]
    if not b1_rejected:
        # 全部通过 → 早退，无需 A2/B2/C
        trace["Final_Output"] = build_final_from_dim_trace(trace["DimTrace"], dim_names)
        trace["Exit_Stage"] = "Consensus_R1"
        return trace

    # ======== A2: 仅修订被 B1 拒绝的维度（通过维度保持不变）========
    rejected_detail = {}
    for d in b1_rejected:
        rejected_detail[d] = {
            "A1_label": get_dim_label(res_a1, d),
            "A1_reasoning": get_dim_reasoning(res_a1, d),
            "B1_feedback": trace["DimTrace"][d]["B1_feedback"]
        }

    a2_instruction = (
        f"Text: \"{text}\"\n\n"
        f"Below are the 6 CSM dimensions. "
        f"PASSED dimensions ({len(dim_names) - len(b1_rejected)}): keep labels unchanged.\n"
        f"FAILED dimensions ({len(b1_rejected)}): REVISE based on B1 feedback.\n\n"
        f"=== B1 FEEDBACK ===\n{_json.dumps(rejected_detail, ensure_ascii=False, indent=2)}\n\n"
        f"=== YOUR A1 OUTPUT ===\n{_json.dumps(res_a1, ensure_ascii=False, indent=2)}\n\n"
        f"Output ALL {len(dim_names)} dimensions. Passed → same label. Failed → corrected."
    )
    res_a2 = await call_annotator(sys_a, a2_instruction)
    trace["Steps"]["A2"] = res_a2

    if not isinstance(res_a2, dict) or "error" in res_a2:
        # A2 失败：回退到 A1 结果
        trace["Final_Output"] = build_final_from_dim_trace(trace["DimTrace"], dim_names)
        trace["Exit_Stage"] = "A2_Error"
        return trace

    for d in dim_names:
        trace["DimTrace"][d]["A2"] = res_a2.get(d, {})

    a2_changed, a2_changes = detect_changed_dimensions(res_a1, res_a2, dim_names)
    trace["DimChanges"]["A1_to_A2"] = {k: v for k, v in a2_changes.items()}

    # 更新被拒且 A2 实际改变了标签的维度
    for d in b1_rejected:
        if d in a2_changed:
            trace["DimTrace"][d]["final_label"] = get_dim_label(res_a2, d)
            trace["DimTrace"][d]["final_source"] = "A2"
            trace["DimTrace"][d]["final_confidence"] = get_dim_confidence(res_a2, d)

    # ======== B2: 仅检查 A2 实际改变了的维度 ========
    if a2_changed:
        b2_focus = {d: {
            "A1_label": a2_changes.get(d, {}).get("from", get_dim_label(res_a1, d)),
            "A2_new_label": a2_changes.get(d, {}).get("to", get_dim_label(res_a2, d)),
            "A2_reasoning": get_dim_reasoning(res_a2, d),
            "B1_feedback": trace["DimTrace"][d]["B1_feedback"]
        } for d in a2_changed}

        b2_instruction = (
            f"Text: \"{text}\"\n\n"
            f"Review ONLY these {len(a2_changed)} revised dimension(s):\n"
            f"{_json.dumps(b2_focus, ensure_ascii=False, indent=2)}\n\n"
            f"Output dimension_feedback per dimension (pass/fail)."
        )
        res_b2 = await call_reviewer(sys_b, b2_instruction)
        trace["Steps"]["B2"] = res_b2

        b2_verdicts = parse_b_verdicts(res_b2, dim_names)
        for d in dim_names:
            v = b2_verdicts.get(d, {"approved": True, "feedback": ""})
            trace["DimTrace"][d]["B2_approved"] = v["approved"]
            trace["DimTrace"][d]["B2_feedback"] = v["feedback"]

        b2_rejected = [d for d in dim_names if not trace["DimTrace"][d]["B2_approved"]]
        if not b2_rejected:
            # B2 全部通过 → 早退
            trace["Final_Output"] = build_final_from_dim_trace(trace["DimTrace"], dim_names)
            trace["Exit_Stage"] = "Consensus_R2"
            return trace

    # ======== C (Arbitrator): 仅仲裁仍有争议的维度 ========
    # 收集两类争议维度：
    #   1. A2 未实际改变但 B1 拒绝了的（Annotator 坚持原判）
    #   2. B2 仍然拒绝的（Reviewer 和 Annotator 僵持）
    still_unresolved = [d for d in dim_names
                        if trace["DimTrace"][d]["final_source"] == "A1"
                        and d in b1_rejected
                        and d not in a2_changed]
    contested = list(set(
        [d for d in dim_names if trace["DimTrace"][d]["final_source"] is None
         or (trace["DimTrace"][d].get("B2_approved") is not None
             and not trace["DimTrace"][d]["B2_approved"])]
        + still_unresolved
    ))

    if contested:
        arb_input = _json.dumps({
            "text": text,
            "contested_dimensions": contested,
            "history": {
                "A1": {d: trace["DimTrace"][d]["A1"] for d in contested if trace["DimTrace"][d]["A1"]},
                "B1_feedback": {d: trace["DimTrace"][d]["B1_feedback"] for d in contested},
                "A2": {d: trace["DimTrace"][d]["A2"] for d in contested if trace["DimTrace"][d]["A2"]},
                "B2_feedback": {d: trace["DimTrace"][d].get("B2_feedback", "") for d in contested}
            }
        }, ensure_ascii=False)
        res_c = await call_arbitrator(sys_c, arb_input)
        trace["Steps"]["C"] = res_c

        if isinstance(res_c, dict) and "error" not in res_c:
            for d in contested:
                c_dim = res_c.get(d, {})
                if c_dim:
                    trace["DimTrace"][d]["C"] = c_dim
                    trace["DimTrace"][d]["final_label"] = get_dim_label(res_c, d)
                    trace["DimTrace"][d]["final_source"] = "C"
                    trace["DimTrace"][d]["final_confidence"] = get_dim_confidence(res_c, d)

    # 构建最终输出并标记退出阶段
    trace["Final_Output"] = build_final_from_dim_trace(trace["DimTrace"], dim_names)
    trace["Exit_Stage"] = "Arbitrated"
    return trace
