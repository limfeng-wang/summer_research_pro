"""
03_六种ABABC排列组合实验（Table 2）
==================================
对应文章 Table 2: 6 种 ABABC 角色分配排列组合的性能对比。

实验设计:
    - 三个模型 (DeepSeek / Doubao / Aliyun) 的全排列 = 6 种组合
    - 每种组合中，模型分别担任 Annotator / Validator / Arbitrator
    - 在 100 条人工标注测试集上运行完整维度级 ABABC 流程
    - 比较 6 种组合的综合评分 (Composite = 0.4*Recall + 0.2*Precision + 0.25*RVS + 0.15*Kappa)

关键设计决策:
    - 维度级 ABABC: 每个维度独立走 A1→B1→A2→B2→C，而非整帖级别的流转
    - 使用 ababc_utils 共享模块的函数（不再是独立副本）
    - 异步并发 (asyncio + aiohttp)，并发限制 10
    - 每个维度留存完整溯源 (final_source: A1/A2/C)

输入:  测试数据.xlsx (100条人工标注)
输出:  ABABC_6Dimensions_Results.xlsx (6个sheet各对应一种排列 + Leaderboard)
       all_results_6dims.json (完整详细结果含reasoning)
"""

import os
import json
import pandas as pd
import numpy as np
import asyncio
import re
import itertools
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, accuracy_score
from sentence_transformers import SentenceTransformer, util
from aiohttp import ClientSession, ClientTimeout
from asyncio import Semaphore
from typing import Dict, List, Tuple, Any
import time
from ababc_utils import (
    parse_b_verdicts as _parse_b_verdicts,
    get_dim_label as _get_dim_label,
    get_dim_confidence as _get_dim_confidence,
    get_dim_reasoning as _get_dim_reasoning,
    detect_changed_dimensions as _detect_changed_dimensions,
    build_dim_trace_empty as _build_dim_trace_empty,
    build_final_from_dim_trace as _build_final_from_dim_trace,
)

# ===================== 1. 配置区域（6维度完整版） =====================

class Config:
    BASE_DIR = r'D:\summer_research\投稿\code_media'
    INPUT_FILE = os.path.join(BASE_DIR, r'all_data\test_data\测试数据.xlsx')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'Permutation_Experiment_Results')
    TRACE_DIR = os.path.join(OUTPUT_DIR, 'traces')

    PROMPT_DIR = os.path.join(BASE_DIR, r'prompts\2_pipeline_roles')
    ROLE_PROMPTS = {
        "Annotator": "annotator_v0.json",
        "Validator": "reviewer_v0.json",
        "Arbitrator": "arbitrator_v0.json"
    }

    MODELS_CONFIG = {
        "DeepSeek": {
            "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "base_url": "https://api.deepseek.com/v1",
            "model_name": "deepseek-chat"
        },
        "Doubao": {
            "api_key": os.getenv("DOUBAO_API_KEY", ""),
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model_name": "doubao-1-5-pro-32k-250115",
            "endpoint_id": os.getenv("DOUBAO_ENDPOINT_ID", "")
        },
        "Aliyun": {
            "api_key": os.getenv("QWEN_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model_name": "qwen-max"
        }
    }

    SBERT_MODEL = r"D:\summer_research\models\paraphrase-multilingual-MiniLM-L12-v2"
    CONCURRENCY_LIMIT = 10
    TIMEOUT = 90
    FORCE_FULL_PIPELINE = False

    # ✅ 修正：完整的6维度配置（与你的金标准列名完全匹配）
    DIMENSIONS = {
        "Perceived Cause": {
            "label_col": "cause_label",
            "reason_col": "cause_reasoning"
        },
        "Symptom Description": {
            "label_col": "symptom_label",
            "reason_col": "symptom_reasoning"
        },
        "Perceived Consequences": {
            "label_col": "consequences_label",
            "reason_col": "consequences_reasoning"
        },
        "Coping and Management": {
            "label_col": "coping_label",
            "reason_col": "coping_reasoning"
        },
        "Emotional Expression": {
            "label_col": "emotion_label",
            "reason_col": "emotion_reasoning"
        },
        "Social Interaction": {
            "label_col": "social_label",
            "reason_col": "social_reasoning"
        }
    }

    TEXT_COLUMN_NAME = "text"

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.TRACE_DIR, exist_ok=True)


# ===================== 2. 数据预处理 =====================

def prepare_data_with_id(df: pd.DataFrame) -> pd.DataFrame:
    """确保DataFrame有唯一ID列"""
    df = df.copy()

    if 'ID' in df.columns:
        df['ID'] = df['ID'].astype(str)
        if df['ID'].duplicated().any():
            print("⚠️  发现重复ID，重新生成...")
            df['ID'] = [f"ROW_{i:04d}" for i in range(len(df))]
    else:
        df['ID'] = [f"ROW_{i:04d}" for i in range(len(df))]

    print(f"✅ 数据ID创建完成：{len(df)} 条样本")
    return df


# ===================== 3. 异步模型管理器 =====================

class AsyncLLMManager:
    """异步多模型管理器。

    支持三种调用方式:
        - DeepSeek / Aliyun: OpenAI 兼容 SDK (AsyncOpenAI)
        - Doubao: 原生 HTTP POST (aiohttp ClientSession)

    内置并发控制 (Semaphore) 和 JSON 安全解析。
    """

    def __init__(self):
        self.clients = {}
        self.sessions = {}
        self.sem = Semaphore(Config.CONCURRENCY_LIMIT)

        for name, cfg in Config.MODELS_CONFIG.items():
            if name == "Doubao":
                self.sessions[name] = ClientSession(
                    headers={"Authorization": f"Bearer {cfg['api_key']}"},
                    timeout=ClientTimeout(total=Config.TIMEOUT)
                )
            else:
                self.clients[name] = AsyncOpenAI(
                    api_key=cfg['api_key'],
                    base_url=cfg['base_url']
                )

    async def call(self, provider: str, system_prompt: str, user_content: str, temperature: float = 0.0) -> Dict:
        async with self.sem:
            try:
                cfg = Config.MODELS_CONFIG[provider]
                if provider == "Doubao":
                    return await self._call_doubao(cfg, system_prompt, user_content, temperature)
                else:
                    return await self._call_openai(provider, system_prompt, user_content, temperature)
            except Exception as e:
                return {"error": str(e), "provider": provider}

    async def _call_openai(self, provider: str, system_prompt: str, user_content: str, temperature: float) -> Dict:
        client = self.clients[provider]
        model = Config.MODELS_CONFIG[provider]['model_name']

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content + "\n\nIMPORTANT: Output valid JSON only."}
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            timeout=Config.TIMEOUT
        )
        return self._safe_parse(response.choices[0].message.content)

    async def _call_doubao(self, cfg: Dict, system_prompt: str, user_content: str, temperature: float) -> Dict:
        session = self.sessions["Doubao"]
        payload = {
            "model": cfg['endpoint_id'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content + " Output valid JSON only."}
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }

        async with session.post(f"{cfg['base_url']}/chat/completions", json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return self._safe_parse(data['choices'][0]['message']['content'])
            else:
                return {"error": f"HTTP {resp.status}"}

    def _safe_parse(self, text: str) -> Dict:
        if not text or not isinstance(text, str):
            return {"error": "empty_response"}
        try:
            clean = re.sub(r'```json\s*|\s*```', '', text.strip())
            return json.loads(clean)
        except:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group()) if match else {"error": "parse_failed"}

    async def close(self):
        for client in self.clients.values():
            await client.close()
        for session in self.sessions.values():
            await session.close()


# ===================== 4. 提示词管理 =====================

def load_prompts() -> Dict[str, str]:
    prompts = {}
    for role, fname in Config.ROLE_PROMPTS.items():
        path = os.path.join(Config.PROMPT_DIR, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                prompts[role] = data.get("system_prompt_template", "")
        else:
            prompts[role] = f"You are a {role}. Output valid JSON."
    return prompts


def build_user_prompt(role: str, context: Dict) -> str:
    text = context.get('text', '')

    if role == 'Annotator':
        if 'history' in context:
            return f"""Task: REVISE annotation based on Validator feedback.
Text: "{text}"

Previous Attempt and Feedback:
{json.dumps(context['history'], ensure_ascii=False, indent=2)}

Instructions: Fix errors explicitly pointed out. Output corrected JSON for ALL 6 dimensions."""
        else:
            return f"""Task: Initial CSM annotation (ALL 6 dimensions).
Text: "{text}"

Annotate ALL 6 dimensions: Perceived Cause, Symptom Description, Perceived Consequences, Coping and Management, Emotional Expression, Social Interaction.
Output valid JSON."""

    elif role == 'Validator':
        prev = json.dumps(context.get('prev', {}), ensure_ascii=False)
        return f"""Task: Verify annotation accuracy for ALL 6 dimensions.
Text: "{text}"

Annotation to Verify:
{prev}

Check each dimension. Output JSON with 'is_correct' (boolean) and specific 'dimension_feedback' (e.g., "Perceived Cause: Hallucination -> set label to 0")."""

    elif role == 'Arbitrator':
        history = json.dumps(context.get('history', {}), ensure_ascii=False)
        return f"""Task: Final arbitration for ALL 6 dimensions.
Text: "{text}"

Complete History (A1, B1, A2, B2):
{history}

Review debate for all dimensions. Output FINAL authoritative JSON with verdict for each dimension."""

    return text


# ===================== 维度级ABABC流程（增强版） =====================

async def run_single_sample(
    sample_id: str,
    text: str,
    assignment: Dict[str, str],
    manager: AsyncLLMManager,
    prompts: Dict[str, str]
) -> Tuple[str, Dict, str, Dict]:
    """
    维度级ABABC流程：6个维度各自走 A1→B1→A2→B2→C，留存全链路
    返回: (sample_id, final_json, exit_stage, trace)
    """
    dim_names = list(Config.DIMENSIONS.keys())
    trace = {
        "ID": sample_id,
        "Steps": {},
        "DimTrace": _build_dim_trace_empty(dim_names),
        "DimChanges": {},
        "Final_Output": {},
        "Exit_Stage": "Unknown"
    }

    # ==== A1: 首次标注 ====
    res_a1 = await manager.call(
        assignment['Annotator'],
        prompts['Annotator'],
        build_user_prompt('Annotator', {'text': text})
    )
    trace["Steps"]["A1"] = res_a1
    if "error" in res_a1:
        return sample_id, res_a1, "Error_A1", trace

    for d in dim_names:
        trace["DimTrace"][d]["A1"] = res_a1.get(d, {})
        trace["DimTrace"][d]["final_label"] = _get_dim_label(res_a1, d)
        trace["DimTrace"][d]["final_source"] = "A1"
        trace["DimTrace"][d]["final_confidence"] = _get_dim_confidence(res_a1, d)

    # ==== B1: 维度级审核 ====
    res_b1 = await manager.call(
        assignment['Validator'],
        prompts['Validator'],
        build_user_prompt('Validator', {'text': text, 'prev': res_a1})
    )
    trace["Steps"]["B1"] = res_b1
    b1_verdicts = _parse_b_verdicts(res_b1, dim_names)

    for d in dim_names:
        v = b1_verdicts.get(d, {"approved": True, "feedback": ""})
        trace["DimTrace"][d]["B1_approved"] = v["approved"]
        trace["DimTrace"][d]["B1_feedback"] = v["feedback"]

    b1_rejected = [d for d in dim_names if not trace["DimTrace"][d]["B1_approved"]]
    if not b1_rejected:
        trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], dim_names)
        trace["Exit_Stage"] = "Consensus_R1"
        return sample_id, trace["Final_Output"], "Consensus_R1", trace

    # ==== A2: 仅修订B1拒绝的维度 ====
    rejected_detail = {}
    for d in b1_rejected:
        rejected_detail[d] = {
            "A1_label": _get_dim_label(res_a1, d),
            "A1_reasoning": _get_dim_reasoning(res_a1, d),
            "B1_feedback": trace["DimTrace"][d]["B1_feedback"]
        }

    a2_instruction = (
        f"Text: \"{text}\"\n\n"
        f"Below are the 6 CSM dimensions. "
        f"Dimensions PASSED: keep their current labels unchanged.\n"
        f"Dimensions FAILED ({len(b1_rejected)}): REVISE based on feedback.\n\n"
        f"{json.dumps(rejected_detail, ensure_ascii=False, indent=2)}\n\n"
        f"Output ALL 6 dimensions. Passed dimensions → same label. "
        f"Failed dimensions → corrected label with improved reasoning."
    )
    res_a2 = await manager.call(
        assignment['Annotator'],
        prompts['Annotator'],
        a2_instruction
    )
    trace["Steps"]["A2"] = res_a2
    if "error" in res_a2:
        trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], dim_names)
        trace["Exit_Stage"] = "A2_Error"
        return sample_id, trace["Final_Output"], "A2_Error", trace

    for d in dim_names:
        trace["DimTrace"][d]["A2"] = res_a2.get(d, {})

    a2_changed, a2_changes = _detect_changed_dimensions(res_a1, res_a2, dim_names)
    trace["DimChanges"]["A1_to_A2"] = a2_changes

    for d in b1_rejected:
        if d in a2_changed:
            trace["DimTrace"][d]["final_label"] = _get_dim_label(res_a2, d)
            trace["DimTrace"][d]["final_source"] = "A2"
            trace["DimTrace"][d]["final_confidence"] = _get_dim_confidence(res_a2, d)

    # ==== B2: 仅检查A2改变了的维度 ====
    b2_rejected = []  # 初始化，防止 a2_changed 为空时引用未定义变量
    if not a2_changed:
        pass  # 无维度被A2改变 → 跳过B2，直接进入C阶段仲裁
    else:
        b2_focus = {d: {
            "A1_label": a2_changes[d]["from"],
            "A2_new_label": a2_changes[d]["to"],
            "A2_reasoning": _get_dim_reasoning(res_a2, d),
            "B1_feedback": trace["DimTrace"][d]["B1_feedback"]
        } for d in a2_changed}

        b2_instruction = (
            f"Text: \"{text}\"\n\n"
            f"Annotator revised {len(a2_changed)} dimension(s). "
            f"Review ONLY these revised dimensions:\n"
            f"{json.dumps(b2_focus, ensure_ascii=False, indent=2)}\n\n"
            f"Determine if each NEW label is correct. "
            f"Output 'dimension_feedback' per dimension indicating pass/fail."
        )
        res_b2 = await manager.call(
            assignment['Validator'],
            prompts['Validator'],
            b2_instruction
        )
        trace["Steps"]["B2"] = res_b2

        b2_verdicts = _parse_b_verdicts(res_b2, dim_names)
        for d in dim_names:
            v = b2_verdicts.get(d, {"approved": True, "feedback": ""})
            trace["DimTrace"][d]["B2_approved"] = v["approved"]
            trace["DimTrace"][d]["B2_feedback"] = v["feedback"]

        b2_rejected = [d for d in dim_names if trace["DimTrace"][d]["B2_approved"] is False]

    if a2_changed and not b2_rejected:
        trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], dim_names)
        trace["Exit_Stage"] = "Consensus_R2"
        return sample_id, trace["Final_Output"], "Consensus_R2", trace

    # ==== C: 仅仲裁仍有争议的维度 ====
    if a2_changed:
        dispute_dims = b2_rejected if b2_rejected else a2_changed
    else:
        dispute_dims = b1_rejected

    c_disputed = {}
    for d in dispute_dims:
        c_disputed[d] = {
            "A1": {"label": _get_dim_label(res_a1, d), "reasoning": _get_dim_reasoning(res_a1, d)},
            "A2": {"label": _get_dim_label(res_a2, d), "reasoning": _get_dim_reasoning(res_a2, d)},
            "B1_feedback": trace["DimTrace"][d]["B1_feedback"],
            "B2_feedback": trace["DimTrace"][d]["B2_feedback"]
        }

    c_instruction = (
        f"Text: \"{text}\"\n\n"
        f"Annotator and Reviewer disagree on {len(dispute_dims)} dimension(s). "
        f"FINAL decision for each:\n"
        f"{json.dumps(c_disputed, ensure_ascii=False, indent=2)}\n\n"
        f"For disputed dims: output final label, reasoning, confidence. "
        f"For undisputed dims: output unchanged from A2."
    )
    res_c = await manager.call(
        assignment['Arbitrator'],
        prompts['Arbitrator'],
        c_instruction
    )
    trace["Steps"]["C"] = res_c

    if "error" not in res_c and isinstance(res_c, dict):
        c_changed, c_changes = _detect_changed_dimensions(res_a2, res_c, dim_names)
        trace["DimChanges"]["A2_to_C"] = c_changes

        for d in dim_names:
            trace["DimTrace"][d]["C"] = res_c.get(d, {})

        for d in dispute_dims:
            if d in c_changed:
                trace["DimTrace"][d]["final_label"] = _get_dim_label(res_c, d)
                trace["DimTrace"][d]["final_source"] = "C"
                trace["DimTrace"][d]["final_confidence"] = _get_dim_confidence(res_c, d)

    # ==== 构建最终输出 ====
    trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], dim_names)

    stages_used = set()
    for d in dim_names:
        stages_used.add(trace["DimTrace"][d]["final_source"])
    if stages_used == {"A1"}:
        trace["Exit_Stage"] = "Consensus_R1"
    elif stages_used == {"A1", "A2"} or stages_used == {"A2"}:
        trace["Exit_Stage"] = "Consensus_R2"
    else:
        trace["Exit_Stage"] = "Arbitrated"

    # DimSummary 方便查看
    trace["DimSummary"] = {
        d: {
            "final_label": trace["DimTrace"][d]["final_label"],
            "final_source": trace["DimTrace"][d]["final_source"],
            "b1_approved": trace["DimTrace"][d]["B1_approved"],
            "b2_approved": trace["DimTrace"][d]["B2_approved"]
        }
        for d in dim_names
    }

    return sample_id, trace["Final_Output"], trace["Exit_Stage"], trace


# ===================== 6. 6维度结果解析与评估（核心修复） =====================

def parse_dimension_results(final_json: Dict) -> Dict[str, Dict]:
    """
    解析JSON中的6维度结果
    ✅ 修复：增加API错误容错
    """
    results = {}

    # 先判断是否有API错误，直接返回默认值
    if "error" in final_json:
        for dim_name in Config.DIMENSIONS.keys():
            results[dim_name] = {
                'label': 0,
                'confidence': 0,
                'reasoning': "",
                'keywords': "",
                'risk': "api_error"
            }
        return results

    for dim_name in Config.DIMENSIONS.keys():
        dim_data = {}

        if isinstance(final_json, dict):
            # 尝试获取维度数据
            dim_obj = final_json.get(dim_name, {})

            if isinstance(dim_obj, dict):
                dim_data['label'] = 1 if str(dim_obj.get("label", "0")) in ['1', 'True', 'true', '是'] else 0
                dim_data['confidence'] = dim_obj.get("confidence", 0)
                dim_data['reasoning'] = str(dim_obj.get("reasoning", ""))
                dim_data['keywords'] = ", ".join(dim_obj.get("keywords", []))
                dim_data['risk'] = dim_obj.get("_risk", "none")
            else:
                # 如果维度不存在或格式错误
                dim_data = {'label': 0, 'confidence': 0, 'reasoning': "", 'keywords': "", 'risk': "missing"}
        else:
            dim_data = {'label': 0, 'confidence': 0, 'reasoning': "", 'keywords': "", 'risk': "parse_error"}

        results[dim_name] = dim_data

    return results


def calculate_dimension_metrics(df_res: pd.DataFrame, dim_name: str, sbert) -> Dict:
    """
    计算单个维度的指标
    ✅ 核心修复：无论是否有金标准，都返回包含所有键的字典（默认值0）
    """
    label_col = f"{dim_name}_label"
    gold_col = Config.DIMENSIONS[dim_name]["label_col"]
    reason_col = f"{dim_name}_reasoning"
    gold_reason_col = Config.DIMENSIONS[dim_name]["reason_col"]

    # 初始化默认指标（避免KeyError）
    metrics = {
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'accuracy': 0.0,
        'kappa': 0.0,
        'rvs': 0.0,
        'composite': 0.0
    }

    # 仅当金标准列和预测列都存在时计算指标
    if gold_col not in df_res.columns or label_col not in df_res.columns:
        return metrics  # 返回默认值，而非空字典

    y_true = df_res[gold_col].fillna(0).astype(int)
    y_pred = df_res[label_col].fillna(0).astype(int)

    # 计算分类指标（增加零除法容错）
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)

    # RVS计算（仅当两者都为正例时）
    rvs_scores = []
    for idx, row in df_res.iterrows():
        if row[gold_col] == 1 and row[label_col] == 1:
            if gold_reason_col in row and reason_col in row:
                pred_r = str(row[reason_col])
                gold_r = str(row[gold_reason_col])
                if pred_r and gold_r and gold_r not in ['0', 'nan', '']:
                    emb = sbert.encode([pred_r, gold_r], convert_to_tensor=True)
                    rvs = float(util.cos_sim(emb[0], emb[1]).item())
                    rvs_scores.append(rvs)

    metrics['rvs'] = np.mean(rvs_scores) if rvs_scores else 0.0
    # RSI综合评分
    metrics['composite'] = (
        metrics['recall'] * 0.4 +
        metrics['precision'] * 0.2 +
        metrics['rvs'] * 0.25 +
        metrics['kappa'] * 0.15
    )

    return metrics


async def run_permutation_safe(
    perm: Tuple[str, str, str],
    df: pd.DataFrame,
    manager: AsyncLLMManager,
    prompts: Dict[str, str],
    sbert
) -> Dict[str, Any]:
    """对一种模型排列组合，在全部测试样本上运行维度级 ABABC 流程。

    流程:
        1. 创建异步任务列表（每帖一个 run_single_sample）
        2. 并发执行全部任务（tqdm 进度条）
        3. 解析每帖 6 维度的标注结果、退出阶段、最终来源
        4. 统计维度流转数据（B1拒绝率、A2修改率、最终来源分布）
        5. 匹配金标准计算匹配率

    Args:
        perm: (Annotator模型, Validator模型, Arbitrator模型) 三元组
        df: 测试数据 DataFrame
        manager: 异步模型管理器
        prompts: {"Annotator": str, "Validator": str, "Arbitrator": str}
        sbert: SBERT 模型实例（用于 RVS 计算）

    Returns:
        {"exp_name": str, "results": list[dict], "exit_stages": dict, "time_cost": float, "dim_transitions": dict}
    """
    annotator, validator, arbitrator = perm
    exp_name = f"A_{annotator[:2]}_V_{validator[:2]}_R_{arbitrator[:2]}".upper()

    assignment = {
        "Annotator": annotator,
        "Validator": validator,
        "Arbitrator": arbitrator
    }

    print(f"\n🔹 开始组合: {exp_name}")
    start_time = time.time()

    # 创建任务
    async def process_row(row):
        sample_id = str(row["ID"])
        result = await run_single_sample(
            sample_id,
            str(row[Config.TEXT_COLUMN_NAME]),
            assignment,
            manager,
            prompts
        )
        return result

    tasks = [process_row(row) for _, row in df.iterrows()]
    results_list = await tqdm.gather(*tasks, desc=f"Processing {exp_name}", total=len(tasks))

    # 构建ID->结果的字典
    results_by_id = {}
    for sample_id, final_json, exit_stage, trace in results_list:
        results_by_id[sample_id] = {
            'final_json': final_json,
            'exit_stage': exit_stage,
            'trace': trace
        }

    # 按原始df顺序处理，解析6维度
    processed_results = []
    exit_stages = {}
    dim_transition_counts = {
        "total_dims": 0,
        "b1_rejected": 0,
        "final_source": {"A1": 0, "A2": 0, "C": 0}
    }

    for _, row in df.iterrows():
        sample_id = str(row["ID"])
        result = results_by_id[sample_id]

        exit_stage = result['exit_stage']
        exit_stages[exit_stage] = exit_stages.get(exit_stage, 0) + 1

        # 解析6维度数据
        dim_results = parse_dimension_results(result['final_json'])

        # From trace: DimSummary
        trace = result.get('trace', {})
        dim_summary = trace.get('DimSummary', {})

        rec = {
            "ID": sample_id,
            "Original_Text": str(row[Config.TEXT_COLUMN_NAME])[:100],
            "Exit_Stage": exit_stage,
            "Experiment": exp_name,
            "Annotator_Model": annotator,
            "Validator_Model": validator,
            "Arbitrator_Model": arbitrator,
        }

        # 添加6维度的详细数据
        for dim_name, dim_data in dim_results.items():
            prefix = dim_name.replace(" ", "_")  # 列名友好化
            rec[f"{prefix}_label"] = dim_data['label']
            rec[f"{prefix}_confidence"] = dim_data['confidence']
            rec[f"{prefix}_reasoning"] = dim_data['reasoning']
            rec[f"{prefix}_keywords"] = dim_data['keywords']
            rec[f"{prefix}_risk"] = dim_data['risk']

            # Dim-level source info
            ds = dim_summary.get(dim_name, {})
            rec[f"{prefix}_final_source"] = ds.get('final_source', 'A1')
            rec[f"{prefix}_b1_approved"] = str(ds.get('b1_approved', 'N/A'))

            # 如果有金标准，计算匹配
            gold_col = Config.DIMENSIONS[dim_name]["label_col"]
            if gold_col in row:
                gold_label = int(row[gold_col]) if pd.notna(row[gold_col]) else 0
                rec[f"{prefix}_gold"] = gold_label
                rec[f"{prefix}_match"] = (dim_data['label'] == gold_label)

        # Accumulate dimension transition stats
        for d_name in Config.DIMENSIONS:
            dim_transition_counts["total_dims"] += 1
            ds = dim_summary.get(d_name, {})
            if ds.get('b1_approved') == False:
                dim_transition_counts["b1_rejected"] += 1
            fs = ds.get('final_source', 'A1')
            if fs in dim_transition_counts["final_source"]:
                dim_transition_counts["final_source"][fs] += 1

        processed_results.append(rec)

    elapsed = time.time() - start_time
    print(f"   ✅ 完成: {len(processed_results)} 样本, 耗时 {elapsed:.1f}s")
    print(f"   📊 退出分布: {exit_stages}")

    print(f"   Dim transitions: B1_rejected={dim_transition_counts['b1_rejected']}")
    print(f"   Final source: A1={dim_transition_counts['final_source']['A1']} A2={dim_transition_counts['final_source']['A2']} C={dim_transition_counts['final_source']['C']}")
    return {
        "exp_name": exp_name,
        "results": processed_results,
        "exit_stages": exit_stages,
        "time_cost": elapsed,
        "dim_transitions": dim_transition_counts
    }


# ===================== 7. 主程序（增加数据有效性检查） =====================

async def main():
    print("🚀 H-RAMOS Permutation Experiment - Full 6 Dimensions")
    print(f"⚡ Concurrency: {Config.CONCURRENCY_LIMIT}")
    print("📊 处理维度: Perceived Cause, Symptom Description, Perceived Consequences, Coping, Emotion, Social")
    print("=" * 80)

    # 1. 加载并预处理数据
    if not os.path.exists(Config.INPUT_FILE):
        print(f"❌ 文件不存在: {Config.INPUT_FILE}")
        return

    df_raw = pd.read_excel(Config.INPUT_FILE)
    print(f"📊 原始数据: {len(df_raw)} 条")
    df = prepare_data_with_id(df_raw)

    # ✅ 优化：细化金标准列检查，输出缺失列和数据有效性
    has_gold = True
    missing_cols = []
    for dim in Config.DIMENSIONS.values():
        for col in [dim['label_col'], dim['reason_col']]:
            if col not in df.columns:
                missing_cols.append(col)
                has_gold = False

    # ✅ 检查金标准列数据有效性（填充空值、统计无效标签）
    if has_gold:
        print("\n🔍 检查金标准列数据有效性...")
        for dim_name, dim_cfg in Config.DIMENSIONS.items():
            label_col = dim_cfg['label_col']
            # 统计空值数量
            null_count = df[label_col].isna().sum()
            # 统计非0/1的值（标签应该只有0/1）
            invalid_vals = df[~df[label_col].isin([0,1,0.0,1.0, '0', '1'])][label_col].count()
            print(f"   {dim_name:25}: 空值={null_count}, 无效标签={invalid_vals}")
            # 填充空值为0（避免计算指标时出错）
            df[label_col] = df[label_col].fillna(0).astype(int)
        print("✅ 检测到所有金标准列，将计算完整指标")
    else:
        print(f"⚠️  缺失金标准列: {missing_cols}，仅计算存在的维度指标")

    # 2. 初始化组件
    print("\n⏳ 加载 SBERT...")
    sbert = SentenceTransformer(Config.SBERT_MODEL)

    print("🔌 初始化异步 API...")
    manager = AsyncLLMManager()
    prompts = load_prompts()

    # 3. 生成排列组合
    model_names = ["DeepSeek", "Doubao", "Aliyun"]
    perms = list(itertools.permutations(model_names, 3))
    print(f"\n⚡ 将运行 {len(perms)} 组实验")

    # 4. 运行所有组合
    all_results = []

    try:
        for perm in perms:
            result = await run_permutation_safe(perm, df, manager, prompts, sbert)
            all_results.append(result)

            # 保存中间结果
            temp_path = os.path.join(Config.OUTPUT_DIR, f"temp_{result['exp_name']}.json")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)

    finally:
        await manager.close()

    # 5. 生成最终报告
    print("\n" + "=" * 80)
    print("📊 生成最终报告（6维度完整版）...")

    output_path = os.path.join(Config.OUTPUT_DIR, "ABABC_6Dimensions_Results.xlsx")
    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    summary_list = []

    for res in all_results:
        exp_name = res['exp_name']
        df_res = pd.DataFrame(res['results'])

        # 只保留核心列到Excel（避免列过多）
        core_cols = ['ID', 'Original_Text', 'Exit_Stage', 'Experiment']
        for dim in Config.DIMENSIONS.keys():
            prefix = dim.replace(" ", "_")
            core_cols.extend([
                f"{prefix}_label",
                f"{prefix}_confidence",
                f"{prefix}_gold",
                f"{prefix}_match"
            ])

        # 过滤存在的列
        existing_cols = [c for c in core_cols if c in df_res.columns]

        # ✅ 修复：Excel sheet名长度限制（最大31字符）
        sheet_name = exp_name[:31] if len(exp_name) > 31 else exp_name
        df_res[existing_cols].to_excel(writer, sheet_name=sheet_name, index=False)

        # 计算6维度的平均指标
        if has_gold:
            dim_metrics = {}
            for dim_name in Config.DIMENSIONS.keys():
                # ✅ 确保每个维度都能拿到完整的metrics字典
                metrics = calculate_dimension_metrics(df_res, dim_name, sbert)
                dim_metrics[dim_name] = metrics

            # ✅ 优化：遍历前先过滤空值，确保计算安全
            avg_metrics = {
                'f1': np.mean([m['f1'] for m in dim_metrics.values() if m['f1'] is not None]),
                'precision': np.mean([m['precision'] for m in dim_metrics.values() if m['precision'] is not None]),
                'recall': np.mean([m['recall'] for m in dim_metrics.values() if m['recall'] is not None]),
                'accuracy': np.mean([m['accuracy'] for m in dim_metrics.values() if m['accuracy'] is not None]),
                'kappa': np.mean([m['kappa'] for m in dim_metrics.values() if m['kappa'] is not None]),
                'rvs': np.mean([m['rvs'] for m in dim_metrics.values() if m['rvs'] is not None]),
                'composite': np.mean([m['composite'] for m in dim_metrics.values() if m['composite'] is not None])
            }

            summary_list.append({
                "Experiment": exp_name,
                "Annotator": res['results'][0]['Annotator_Model'],
                "Validator": res['results'][0]['Validator_Model'],
                "Arbitrator": res['results'][0]['Arbitrator_Model'],
                "Avg_Accuracy": round(avg_metrics['accuracy'], 4),
                "Avg_F1": round(avg_metrics['f1'], 4),
                "Avg_Precision": round(avg_metrics['precision'], 4),
                "Avg_Recall": round(avg_metrics['recall'], 4),
                "Avg_Kappa": round(avg_metrics['kappa'], 4),
                "Avg_RVS": round(avg_metrics['rvs'], 4),
                "Composite": round(avg_metrics['composite'], 4),
                "Time_Cost(s)": round(res['time_cost'], 1),
                **{f"Exit_{k}": v for k, v in res['exit_stages'].items()}
            })

            # 打印各维度详情
            print(f"\n📈 {exp_name}:")
            for dim_name, metrics in dim_metrics.items():
                print(f"   {dim_name:25}: F1={metrics['f1']:.3f}, RVS={metrics['rvs']:.3f}")

    # 排行榜
    if summary_list:
        df_summary = pd.DataFrame(summary_list).sort_values(by="Composite", ascending=False)
        df_summary.to_excel(writer, sheet_name="Leaderboard", index=False)
        print("\n🏆 综合排行榜 (6维度平均):")
        print(df_summary[['Experiment', 'Composite', 'Avg_F1', 'Avg_RVS', 'Exit_Consensus_R1']].to_string(index=False))

    writer.close()

    # 保存完整JSON（包含所有维度的reasoning等详细信息）
    json_path = os.path.join(Config.OUTPUT_DIR, "all_results_6dims.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'total_samples': len(df),
                'dimensions': list(Config.DIMENSIONS.keys()),
                'id_generation': 'auto_index'
            },
            'results': all_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！")
    print(f"📊 Excel: {output_path}")
    print(f"🔍 JSON: {json_path}")
    print(f"📝 每个样本包含6个维度的完整标注信息")


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()