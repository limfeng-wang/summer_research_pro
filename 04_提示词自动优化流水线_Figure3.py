"""
04_提示词自动优化流水线（Figure 3）
=================================
对应文章 Figure 3: Meta-LLM 驱动的提示词自动优化流水线。

核心机制（详见 核心机制原理说明.md 第一节）:
    Meta-LLM (deepseek-chat) 每轮收到完整诊断报告 → 分析瓶颈 → 改写提示词。
    不是模型在学习，而是"每次拿到更新的诊断报告，尝试不同的改写策略"。

优化循环（每轮）:
    1. 用 current_prompts 在 100 条测试集上跑完整维度级 ABABC
    2. 计算 System Final 指标 (Kappa, F1, Recall, Precision)
    3. 分析各阶段贡献 (F1_A1_only → +A2 → +C_full)
    4. 维度级过渡统计 (B1拒绝率, A2修改率, B2再拒率, C使用率)
    5. 收集维度级错误案例 (FN/FP/FalseReject/MissedReject)
    6. 计算提示词 diff (本轮 vs 上轮, 本轮 vs Champion)
    7. 打包以上全部数据发给 Meta-LLM → 生成改写后的提示词
    8. 早停判断: patience=3 无提升 / Kappa≥0.85 / max_rounds=20

基线对比（审稿人 Comment 32 回复）:
    - Round 0a: 零提示词基线 (无 system prompt)
    - Round 0b: 单句提示词基线 (one-shot minimal instruction)
    - Round 0c: 专家 v0 提示词 (优化起点)

早停与回滚:
    - 连续 3 轮无 >0.5% 提升 → 停止
    - 当前 Kappa < 历史最佳 - 0.01 → 智能回滚到 Champion Prompt
    - Kappa ≥ 0.85 → 达标停止
    - 停止后自动部署 best_prompts → best_prompts_final/ (供 05 直接加载)

输入:  测试数据.xlsx (100条), prompts/2_pipeline_roles/*.json (v0基线)
       merged_dictionary_v2.json (动态词典)

输出:  best_prompts_final/{Annotator,Reviewer,Arbitrator}_best.txt (→ 05)
       experiment_full_trace/Round_N_SystemReport.json (每轮完整报告)
       prompts_history/{role}_v{N}.txt (每轮提示词存档)
       best_prompts/{role}_BEST_R{N}.txt (Champion Prompt 快照)
       00_baseline_comparison.xlsx (零提示词 vs 单句 vs 专家基线)
"""

import os
import json
import pandas as pd
import numpy as np
import asyncio
import re
import shutil
import hashlib
import time
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, accuracy_score
from sentence_transformers import SentenceTransformer, util
from aiohttp import ClientSession, ClientTimeout
from asyncio import Semaphore
from typing import Dict, List, Any, Tuple
from ababc_utils import (
    parse_b_verdicts as _parse_b_verdicts,
    get_dim_label as _get_dim_label,
    get_dim_confidence as _get_dim_confidence,
    get_dim_reasoning as _get_dim_reasoning,
    detect_changed_dimensions as _detect_changed_dimensions,
    build_dim_trace_empty as _build_dim_trace_empty,
    build_final_from_dim_trace as _build_final_from_dim_trace,
)

# ===================== 1. 全局配置与基础设施 (Infrastructure) =====================

class Config:
    # --- 基础路径 ---
    BASE_ROOT = r'D:\summer_research\投稿\code_media'
    INPUT_FILE = os.path.join(BASE_ROOT, r'all_data\test_data\测试数据.xlsx')
    LEXICON_FILE = os.path.join(BASE_ROOT, r'co-occurrence network\词典存储\merged_dictionary_v2.json')

    # ✅ 新增：TRACE_ROOT 定义
    TRACE_ROOT = os.path.join(BASE_ROOT, r'experiment_traces')
    BASELINE_PROMPT_DIR = os.path.join(BASE_ROOT, r'prompts\2_pipeline_roles')

    # 自动创建目录
    HISTORY_ROOT = os.path.join(BASE_ROOT, r'prompts_history')  # 存 Prompt 版本
    LOG_ROOT = os.path.join(BASE_ROOT, r'experiment_full_trace')  # 存全链路日志
    BEST_PROMPT_DIR = os.path.join(BASE_ROOT, r'best_prompts')  # 存最佳提示词

    # ✅ 新增：早停与优化目标配置
    STOPPING_CRITERIA = {
        "target_kappa": 0.85,        # 核心指标目标：Kappa ≥ 0.85
        "target_f1": 0.90,           # 备用指标：F1 ≥ 0.90
        "patience": 3,               # 连续3轮无提升则停
        "min_delta": 0.005,          # 提升小于0.5%视为停滞
        "max_rounds": 20             # 硬上限
    }

    ROLE_WEIGHTS = {
        "Annotator": {  # 高召回导向
            "recall": 0.40,
            "precision": 0.30,
            "rvs": 0.15,
            "kappa": 0.15
        },
        "Reviewer": {  # 高精确导向
            "recall": 0.20,
            "precision": 0.50,
            "rvs": 0.15,
            "kappa": 0.15
        },
        "Arbitrator": {  # 综合平衡导向
            "recall": 0.35,
            "precision": 0.35,
            "rvs": 0.15,
            "kappa": 0.15
        }
    }

    # --- API Keys (请填入真实 Key) ---
    KEYS = {
        "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
        "doubao": os.getenv("DOUBAO_API_KEY", ""),
        "qwen": os.getenv("QWEN_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
    }

    # 豆包配置 (Endpoint ID)
    DOUBAO_EP_ID = os.getenv("DOUBAO_ENDPOINT_ID", "")

    # --- 模型路由 ---
    WORKER_MODELS = {
        "Annotator": "deepseek-chat",
        "Reviewer": "doubao-1-5-pro-32k-250115",
        "Arbitrator": "qwen-max"
    }
    META_MODEL = "deepseek-chat"

    # --- 运行参数 ---
    CONCURRENCY_LIMIT = 10
    TIMEOUT = 120
    SBERT_PATH = r"D:\summer_research\models\paraphrase-multilingual-MiniLM-L12-v2"

    OPTIMIZATION_ROUNDS = 10
    FORCE_FULL_PIPELINE = False

    # --- 维度映射 ---
    DIM_MAP = {
        "Perceived Cause": "cause",
        "Symptom Description": "symptom",
        "Perceived Consequences": "consequences",
        "Coping and Management": "coping",
        "Emotional Expression": "emotion",
        "Social Interaction": "social"
    }
    DEBUG_MODE = False


# ===================== 2. 提示词工程结构 (Prompt Engineering) =====================

# 2.1 Meta-LLM优化模板: 系统最终目标导向版本
# 2.1 Meta-LLM Optimization Template: Concise & Refactoring-Oriented Version
META_SYSTEM_PROMPT = """
# Role & Mission
You are the **System Architect** for a Multi-Agent NLP pipeline.
**PRIMARY GOAL**: Maximize SYSTEM FINAL OUTPUT quality (Kappa ≥ 0.85, F1 ≥ 0.90).
**CRITICAL CONSTRAINT**: **Prompt Pruning & Refactoring**. A bloated prompt (e.g., >4500 chars) leads to model distraction and is a failure condition.

# =============================================================================
# 🎯 PRUNING & REFACTORING RULES (MANDATORY)
# =============================================================================
1. **EXAMPLE QUOTA**: Each dimension MUST have **NO MORE THAN 2** highly representative examples. When adding a new example, you MUST delete an old one.
2. **REFACTOR, DON'T APPEND**: Do not just add new rules to fix edge cases. Instead, **rewrite the core definition** of the dimension to inherently cover the new case.
3. **DELETE REDUNDANCY**: Scan the prompt for repetitive constraints. If two sentences convey similar logic, merge them into one concise statement.
4. **LOGIC DENSITY**: Use high-density language. Instead of "You should try to find if there are any indications of...", use "Detect...".

# =============================================================================
# 🎯 DIAGNOSE → STRATEGIZE → EXECUTE (Work in this order)
# =============================================================================

## STEP 1: DIAGNOSE — Read the error statistics and identify the TOP problem dimension
- Which dimension has the MOST errors? (FP + FN + LowRVS)
- What is the dominant ERROR TYPE for that dimension? (Hallucination/FP vs Missed Signal/FN)
- Is the problem in ANNOTATOR (wrong initial label) or REVIEWER (wrong review decision)?

## STEP 2: STRATEGIZE — Pick ONE concrete strategy based on diagnosis

**If dominant error is FP (Hallucination) on dimension X:**
→ Tighten the INCLUDE criteria for dimension X. Add a "NEGATIVE EXAMPLE" showing what does NOT count.
→ Example: If Social Interaction has 12 FP → show cases like "歯が痛い‼️‼️‼️" that are Emotional Expression, NOT Social.
→ Example: If Perceived Consequences has FP → show that "疼死了" is pain intensity (Symptom), not functional impact (Consequence).

**If dominant error is FN (Missed Signal) on dimension X:**
→ Broaden the INCLUDE criteria for dimension X. Add an "IMPLICIT SIGNAL" example.
→ Example: If Perceived Consequences has 8 FN → explain that "睡不着" (can't sleep) is a consequence even if not explicitly stated as caused by pain.
→ Example: If Coping has FN → explain that mentioning a drug name (布洛芬) alone qualifies as coping behavior.

**If Reviewer is the bottleneck (False Rejection or Missed Rejection):**
→ Align Reviewer's audit standards with the Annotator's updated criteria.
→ Add dimension-specific audit rules to the Reviewer prompt.

**If C-Arbitrator is overused (deadlock rate > 5%):**
→ The Annotator and Reviewer definitions are contradictory for the deadlocked dimension.
→ Rewrite that dimension's criteria to have a CLEAR, SINGLE interpretation.

## STEP 3: EXECUTE — Apply the strategy
- Modify ONLY the relevant dimension's criteria section. Do not touch dimensions that are performing well.
- After making changes, document what you changed and why.

# =============================================================================
# 🚨 TIER 0: ZERO-TOLERANCE CONSTRAINTS
# =============================================================================
1. SACRED PLACEHOLDER: Preserve `{lexicon}` EXACTLY as-is (Annotator only).
2. XML FORMAT MANDATORY: Wrap ENTIRE output in <H_RAMOS_PROMPT_V1>...</H_RAMOS_PROMPT_V1>
3. SCHEMA IMMUTABLE: Never modify JSON keys ("label", "confidence", "reasoning", "keywords", "_meta")

# =============================================================================
# 📋 CHANGE DOCUMENTATION (REQUIRED in your output)
# =============================================================================
Before the prompt, document:
- **Diagnosed Problem**: [Which dimension + error type + count]
- **Strategy Applied**: [Which strategy from Step 2]
- **Specific Change**: [Which section of the prompt was modified and how]
"""

# 2.2 角色诊断映射
# 2.2 Role Diagnosis Map
# 2.2 Role Diagnosis Map
LOGIC_DIAGNOSIS_MAP = {
    "Annotator": """
    **OBJECTIVE: High-Sensitivity / Recall**
    - **Action**: If Recall is low, broaden dimension boundaries and lower 'certainty' barriers.
    - **Pruning**: Remove specific 'include/exclude' lists if they are too long; replace with abstract category definitions.
    - **Constraint**: Ensure `{lexicon}` is present. Max 2 examples per dimension.
    """,

    "Reviewer": """
    **OBJECTIVE: High-Precision / Error Gating**
    - **Action**: If Precision is low, enforce 'Direct Evidence' check.
    - **Action**: If 'False Pass' rate is high, instruct Reviewer to be 'Skeptical' of Annotator's confidence.
    - **Pruning**: Remove vague instructions like 'Check for general quality'. Focus on specific logical fallacies.
    """,

    "Arbitrator": """
    **OBJECTIVE: Deadlock Resolution**
    - **Action**: Refine tie-breaking priority (Precision vs Recall based on system needs).
    - **Constraint**: Output format must perfectly match Annotator schema.
    - **Pruning**: Keep reasoning minimal. Focus on the verdict for the disputed dimension only.
    """
}
# 2.3 核心提示词库
DEFAULT_PROMPTS = {
    "Annotator": "",
    "Reviewer": "",
    "Arbitrator": ""
}

class PromptLibrary:
    META_SYSTEM_PROMPT = META_SYSTEM_PROMPT
    DIAGNOSIS_MAP = LOGIC_DIAGNOSIS_MAP
    DEFAULT_PROMPTS = DEFAULT_PROMPTS


class LexiconManager:
    """动态词典管理器：加载、查询、更新"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.lexicon = self._load()
        self.new_words_buffer = {dim: set() for dim in Config.DIM_MAP}

    def _load(self):
        """加载词典（若不存在则复制种子）"""
        if not os.path.exists(self.filepath):
            seed_path = os.path.join(os.path.dirname(self.filepath), 'lexicon_seed.json')
            if os.path.exists(seed_path):
                shutil.copy(seed_path, self.filepath)
            else:
                return {dim: [] for dim in Config.DIM_MAP}

        with open(self.filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_lexicon_for_prompt(self):
        """格式化为提示词字符串"""
        return json.dumps(self.lexicon, ensure_ascii=False, indent=2)

    def check_and_buffer_new_words(self, final_output):
        """检查新词并加入缓冲区"""
        if not isinstance(final_output, dict):
            return

        for dim_full, dim_data in final_output.items():
            if not isinstance(dim_data, dict) or "keywords" not in dim_data:
                continue

            existing_words = set(self.lexicon.get(dim_full, []))
            for keyword in dim_data.get("keywords", []):
                if keyword not in existing_words and keyword not in self.new_words_buffer[dim_full]:
                    self.new_words_buffer[dim_full].add(keyword)
                    if Config.DEBUG_MODE:
                        print(f"🆕 New word detected: [{dim_full}] {keyword}")

    def commit_new_words(self, round_i):
        """将缓冲区新词正式加入词典并保存"""
        total_new = sum(len(words) for words in self.new_words_buffer.values())
        if total_new == 0:
            if Config.DEBUG_MODE:
                print("ℹ️ 本轮未发现新词")
            return 0

        # 将新词加入词典
        for dim_full, new_words in self.new_words_buffer.items():
            if dim_full not in self.lexicon:
                self.lexicon[dim_full] = []
            self.lexicon[dim_full].extend(sorted(new_words))
            self.lexicon[dim_full] = sorted(list(set(self.lexicon[dim_full])))

        # 备份旧版本
        backup_path = f"{self.filepath}.backup_round_{round_i}"
        shutil.copy(self.filepath, backup_path)

        # 保存新版本
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.lexicon, f, ensure_ascii=False, indent=2)

        # 清空缓冲区
        self.new_words_buffer = {dim: set() for dim in Config.DIM_MAP}

        print(f"📚 词典已更新: +{total_new} 个新词，备份: {backup_path}")
        return total_new

    def get_stats(self):
        """统计词典状态"""
        return {dim: len(words) for dim, words in self.lexicon.items()}


# ===================== 3. 记忆与日志层 (Memory & Logging) =====================

class OptimizationMemory:
    """记忆模块：记录系统最终指标和各角色表现 + 提示词变更追踪"""

    def __init__(self):
        self.history = {"Annotator": [], "Reviewer": [], "Arbitrator": []}
        self.system_history = []
        self.prompt_versions = {"Annotator": {}, "Reviewer": {}, "Arbitrator": {}}
        self.best_system_score = 0.0
        self.best_round = 0
        # 新增: 变更日志追踪
        self.change_log = {"Annotator": [], "Reviewer": [], "Arbitrator": []}

    def record_system(self, round_i, system_metrics):
        """记录系统最终指标"""
        prev_score = self.system_history[-1]['kappa'] if self.system_history else 0.0
        delta = system_metrics['Kappa'] - prev_score

        self.system_history.append({
            "round": round_i,
            "kappa": system_metrics['Kappa'],
            "f1": system_metrics['F1'],
            "recall": system_metrics['Recall'],
            "precision": system_metrics['Precision'],
            "delta": delta
        })

        # 更新最佳记录
        if system_metrics['Kappa'] > self.best_system_score:
            self.best_system_score = system_metrics['Kappa']
            self.best_round = round_i

        return delta

    def record(self, role, round_i, metrics, prompt_content):
        """记录角色指标"""
        prev_score = 0
        if len(self.history[role]) > 0:
            prev_score = self.history[role][-1]['score']

        delta = metrics['Composite_Score'] - prev_score if prev_score > 0 else 0

        self.history[role].append({
            "round": round_i,
            "score": metrics['Composite_Score'],
            "delta": delta,
            "metrics_detail": metrics
        })
        self.prompt_versions[role][round_i] = {
            "prompt": prompt_content,
            "score": metrics['Composite_Score']
        }

    def get_system_trend(self, n_rounds=3):
        """获取系统指标趋势"""
        if len(self.system_history) < 2:
            return "Insufficient system history."

        recent = self.system_history[-n_rounds:]
        kappas = [r['kappa'] for r in recent]

        trend = "STABLE"
        if all(r['delta'] > 0.005 for r in recent[1:]):
            trend = "IMPROVING ↗️"
        elif all(r['delta'] < -0.005 for r in recent[1:]):
            trend = "DETERIORATING ↘️"

        return f"""
=== SYSTEM FINAL METRIC TREND (Last {len(recent)} rounds) ===
Kappa: {' -> '.join([f'{k:.3f}' for k in kappas])}
Target: {Config.STOPPING_CRITERIA['target_kappa']}
Best: Round {self.best_round} (Kappa: {self.best_system_score:.4f})
Trend: {trend}
"""

    def get_last_change_details(self, role):
        """获取上一轮修改详情"""
        if len(self.history[role]) < 2:
            return "No previous round available."

        last_round = len(self.history[role])
        prev_prompt = self.prompt_versions[role].get(last_round - 1, {}).get("prompt", "")
        curr_prompt = self.prompt_versions[role].get(last_round, {}).get("prompt", "")

        if prev_prompt == curr_prompt:
            return "⚠️ WARNING: Prompt unchanged from previous round."

        return f"""
=== PROMPT EVOLUTION ===
Previous Round {last_round - 1} Score: {self.history[role][-2]['score']:.4f}
Current Round {last_round} Score: {self.history[role][-1]['score']:.4f}
Change Delta: {self.history[role][-1]['delta']:+.4f}
"""

    def get_history_warning(self, role):
        """如果上一轮是负优化，生成警告"""
        if not self.history[role]:
            return ""

        last = self.history[role][-1]
        if last['delta'] < -0.01:
            return f"\n⚠️ HISTORY WARNING: Round {last['round']} REGRESSION (Score dropped by {abs(last['delta']):.4f})."
        return ""

    def compute_prompt_diff(self, role, round_i):
        """计算本轮提示词相对于上一轮的具体变更（解决Meta-LLM不知改了什么的问题）"""
        if round_i <= 1:
            return "First round — no previous prompt to compare."
        prev = self.prompt_versions[role].get(round_i - 1, {}).get("prompt", "")
        curr = self.prompt_versions[role].get(round_i, {}).get("prompt", "")
        if not prev or not curr:
            return "Cannot compute diff — missing prompt version."

        import difflib
        prev_lines = prev.splitlines(keepends=True)
        curr_lines = curr.splitlines(keepends=True)
        differ = difflib.unified_diff(prev_lines, curr_lines,
                                       fromfile=f'Round_{round_i-1}',
                                       tofile=f'Round_{round_i}',
                                       lineterm='')
        diff_output = ''.join(differ)
        if not diff_output.strip():
            return "⚠️ WARNING: Prompt UNCHANGED from previous round — optimization had no effect."

        # 截断过长的diff
        if len(diff_output) > 3000:
            diff_output = diff_output[:3000] + "\n... (diff truncated, full version in history)"

        # 记录到变更日志
        score_delta = 0.0
        if len(self.system_history) >= round_i:
            score_delta = self.system_history[round_i - 1].get('delta', 0.0)
        self.change_log[role].append({
            "round": round_i,
            "delta_kappa": score_delta,
            "diff_summary": diff_output[:500]
        })

        return f"""
=== PROMPT DIFF (Round {round_i-1} → Round {round_i}) ===
Kappa change: {score_delta:+.4f}
{'+' if score_delta > 0 else '' if score_delta == 0 else '⚠️ '}Lines changed:
{diff_output}
"""

    def get_failed_approaches(self, role):
        """汇总该角色历史上所有失败的尝试，防止Meta-LLM重蹈覆辙"""
        failures = [c for c in self.change_log[role] if c['delta_kappa'] < -0.002]
        if not failures:
            return ""
        lines = ["\n⚠️ FAILED APPROACH HISTORY (DO NOT REPEAT):"]
        for f in failures:
            lines.append(f"  Round {f['round']}: Δ={f['delta_kappa']:+.4f}")
        return "\n".join(lines)


class ResultLogger:
    """全链路留痕"""

    @staticmethod
    def save_trace(round_i, trace_data, suffix=""):
        """旧版：将所有数据存入单个JSON文件"""
        df = pd.DataFrame(trace_data)
        file_path = os.path.join(Config.LOG_ROOT, f"Round_{round_i}_Full_Trace{suffix}.json")
        df.to_json(file_path, orient='records', force_ascii=False, indent=2)
        print(f"📄 Trace Log Saved: {file_path}")

    @staticmethod
    def save_trace_per_model(round_i, trace_results):
        """新版：为每个模型的每次输出创建独立文件"""
        os.makedirs(Config.TRACE_ROOT, exist_ok=True)
        base_dir = os.path.join(Config.TRACE_ROOT, f"Round_{round_i}")

        for role_stage in ["A1", "B1", "A2", "B2", "C"]:
            os.makedirs(os.path.join(base_dir, role_stage), exist_ok=True)

        for trace in trace_results:
            row_id = trace["ID"]

            for stage, data in trace["Steps"].items():
                if stage not in ["A1", "B1", "A2", "B2", "C"]:
                    continue

                stage_dir = os.path.join(base_dir, stage)
                file_path = os.path.join(stage_dir, f"{row_id}.json")

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "ID": row_id,
                        "stage": stage,
                        "exit_stage": trace.get("Exit_Stage"),
                        "output": data
                    }, f, ensure_ascii=False, indent=2)

        print(f"📁 Trace logs saved to: {base_dir}")


# ===================== 4. 底层通信模块 (AsyncLLM) =====================

class AsyncLLMManager:
    def __init__(self):
        self.clients = {}
        self.request_times = []
        self.max_rpm = 20

        # DeepSeek & Qwen
        if Config.KEYS["deepseek"]:
            self.clients["deepseek-chat"] = AsyncOpenAI(
                api_key=Config.KEYS["deepseek"],
                base_url="https://api.deepseek.com/v1"
            )
        if Config.KEYS["qwen"]:
            self.clients["qwen-max"] = AsyncOpenAI(
                api_key=Config.KEYS["qwen"],
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

        # Doubao HTTP Session
        self.doubao_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.sem = Semaphore(Config.CONCURRENCY_LIMIT)
        self.session = ClientSession(
            headers={"Authorization": f"Bearer {Config.KEYS['doubao']}"},
            timeout=ClientTimeout(total=Config.TIMEOUT)
        )

    async def _check_rate_limit(self):
        """速率限制检查"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) >= self.max_rpm:
            wait_time = 30 - (now - self.request_times[0]) + 1
            if wait_time > 0:
                print(f"  ⏳ 速率保护: 等待{wait_time:.1f}秒...")
                await asyncio.sleep(wait_time)

    async def call(self, model_key, messages, temperature=0.0):
        # 确保JSON指令存在
        has_json_instruction = any(
            "json" in str(msg.get("content", "")).lower()
            for msg in messages
        )

        if not has_json_instruction and messages:
            messages[0]["content"] = messages[0].get("content", "") + "\n\nIMPORTANT: Output valid JSON only, no markdown code blocks."

        async with self.sem:
            try:
                await self._check_rate_limit()

                if model_key == "doubao-1-5-pro-32k-250115":
                    raw_response = await self._call_doubao(messages, temperature)
                else:
                    client = self.clients.get(model_key, self.clients.get("deepseek-chat"))
                    resp = await client.chat.completions.create(
                        model=model_key,
                        messages=messages,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        timeout=Config.TIMEOUT
                    )
                    raw_response = resp.choices[0].message.content
                    self.request_times.append(time.time())

                return self._safe_json_parse(raw_response, model_key)

            except Exception as e:
                print(f"  [API Error] {model_key}: {e}")
                return {"error": "api_failed", "raw": str(e)}

    def _safe_json_parse(self, text, role_name):
        """增强版JSON净化器"""
        if isinstance(text, dict):
            return text

        if not text or not isinstance(text, str):
            return {"error": "empty_or_invalid_response", "details": f"Expected str, got {type(text)}"}

        cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', text.strip())

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            else:
                return {"error": "not_dict", "raw": str(parsed)[:200]}
        except json.JSONDecodeError:
            pass

        # 使用栈匹配法提取最外层JSON对象
        def extract_json_objects(s):
            objects = []
            stack = []
            in_string = False
            escape = False
            start = -1

            for i, char in enumerate(s):
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False

                if not in_string:
                    if char == '{':
                        if not stack:
                            start = i
                        stack.append(char)
                    elif char == '}':
                        if stack:
                            stack.pop()
                            if not stack and start != -1:
                                objects.append(s[start:i+1])
                                start = -1
            return objects

        json_candidates = extract_json_objects(cleaned)

        for candidate in sorted(json_candidates, key=len, reverse=True):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        return {
            "error": "json_parse_failed",
            "raw_snippet": cleaned[:200] + "..." if len(cleaned) > 200 else cleaned
        }

    async def _call_doubao(self, messages, temperature):
        """豆包HTTP调用"""
        msg_copy = [msg.copy() for msg in messages]

        if msg_copy and "json" not in msg_copy[0].get('content', '').lower():
            msg_copy[0]['content'] += " Output valid JSON only, no markdown."

        payload = {
            "model": Config.DOUBAO_EP_ID,
            "messages": msg_copy,
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }

        async with self.session.post(self.doubao_url, json=payload) as resp:
            self.request_times.append(time.time())

            if resp.status == 200:
                try:
                    data = await resp.json()
                    content = data['choices'][0]['message']['content']
                    return content
                except Exception as e:
                    return json.dumps({"error": "doubao_parse_fail", "details": str(e)})
            else:
                error_detail = await resp.text()
                return json.dumps({
                    "error": "doubao_http_error",
                    "status_code": resp.status,
                    "details": error_detail[:200]
                })

    async def close(self):
        """关闭 HTTP 会话"""
        if hasattr(self, 'session') and not self.session.closed:
            await self.session.close()
            print("🔒 AsyncLLMManager session closed")


# ===================== 5. 评估核心层 (维度级) =====================

class Evaluator:
    def __init__(self):
        print("⏳ Loading SBERT...")
        self.sbert = SentenceTransformer(Config.SBERT_PATH)

    def calculate_rvs(self, p_reason, g_reason):
        """计算推理向量相似度"""
        if not p_reason or not g_reason:
            return 0.0
        try:
            embeddings = self.sbert.encode([str(p_reason), str(g_reason)], convert_to_tensor=True)
            return util.cos_sim(embeddings[0], embeddings[1]).item()
        except Exception as e:
            return 0.0

    # ========== 维度级指标提取 ==========

    def _dim_labels_from_trace(self, trace, dim_full, stage="Final"):
        """从DimTrace提取某个维度的标签，始终返回 0 或 1"""
        dt = trace.get("DimTrace", {}).get(dim_full, {})
        if stage == "Final":
            val = dt.get("final_label")
            return val if val in (0, 1) else 0
        elif stage == "A1":
            return _get_dim_label(trace.get("Steps", {}).get("A1", {}), dim_full)
        elif stage == "A2":
            return _get_dim_label(trace.get("Steps", {}).get("A2", {}), dim_full)
        elif stage == "C":
            return _get_dim_label(trace.get("Steps", {}).get("C", {}), dim_full)
        return 0

    def extract_labels_and_reasons(self, trace_results: List[Dict], df_gold: pd.DataFrame) -> Tuple:
        """
        维度级标签和推理提取。
        现在用 DimTrace 确定每个维度的最终来源。
        """
        data_by_stage = {
            "A1": {"y_true": [], "y_pred": [], "reasons": []},
            "A2": {"y_true": [], "y_pred": [], "reasons": []},
            "Final": {"y_true": [], "y_pred": [], "reasons": []}
        }

        for res in trace_results:
            row_id = str(res["ID"])
            gold_row = df_gold[df_gold["ID"].astype(str) == row_id]
            if gold_row.empty:
                continue
            gold_row = gold_row.iloc[0]

            steps = res.get("Steps", {})
            a1_output = steps.get("A1", {})
            a2_output = steps.get("A2", {})

            for dim_full, dim_pre in Config.DIM_MAP.items():
                g_label = int(gold_row.get(f"{dim_pre}_label", 0))
                g_reason = str(gold_row.get(f"{dim_pre}_reasoning", ""))

                # A1 标签（所有维度都有A1）
                a1_label = self._dim_labels_from_trace(res, dim_full, "A1")
                a1_reason = _get_dim_reasoning(a1_output, dim_full)

                # A2 标签（仅在被读取时有意义）
                a2_label = self._dim_labels_from_trace(res, dim_full, "A2")
                a2_reason = _get_dim_reasoning(a2_output, dim_full) if a2_output else ""

                # Final 标签（用 DimTrace 的 final_label）
                final_label = self._dim_labels_from_trace(res, dim_full, "Final")
                final_source = res.get("DimTrace", {}).get(dim_full, {}).get("final_source", "A1")
                # 根据final_source取对应的reasoning
                if final_source == "A1":
                    final_reason = a1_reason
                elif final_source == "A2":
                    final_reason = a2_reason
                elif final_source == "C":
                    c_output = steps.get("C", {})
                    final_reason = _get_dim_reasoning(c_output, dim_full)
                else:
                    final_reason = a1_reason

                # 添加到各阶段
                for stage, label, reason in [("A1", a1_label, a1_reason),
                                              ("A2", a2_label, a2_reason),
                                              ("Final", final_label, final_reason)]:
                    data_by_stage[stage]["y_true"].append(g_label)
                    data_by_stage[stage]["y_pred"].append(label)
                    if g_label == 1 and label == 1:
                        data_by_stage[stage]["reasons"].append((reason, g_reason))

        return data_by_stage

    def compute_system_final_metrics(self, trace_results: List[Dict], df_gold: pd.DataFrame) -> Dict:
        """
        基于维度级 DimTrace 计算系统最终指标。
        每个维度使用其 final_source 决定用哪一阶段的输出。
        """
        y_true, y_pred = [], []

        for res in trace_results:
            row_id = str(res["ID"])
            gold_row = df_gold[df_gold["ID"].astype(str) == row_id]
            if gold_row.empty:
                continue
            gold_row = gold_row.iloc[0]

            for dim_full, dim_pre in Config.DIM_MAP.items():
                g_label = int(gold_row.get(f"{dim_pre}_label", 0))
                p_label = self._dim_labels_from_trace(res, dim_full, "Final")

                y_true.append(g_label)
                y_pred.append(p_label)

        if not y_true:
            return {"Recall": 0.0, "Precision": 0.0, "F1": 0.0, "Kappa": 0.0, "Accuracy": 0.0, "Macro_F1": 0.0, "Support": 0}

        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        return {
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "Kappa": kappa,
            "Accuracy": accuracy,
            "Macro_F1": macro_f1,
            "Support": len(y_true)
        }

    # ========== 维度级过渡统计（核心新增） ==========

    def compute_dimension_transitions(self, trace_results: List[Dict]) -> Dict:
        """
        计算维度级的 pipeline 过渡统计。
        这是你提到的"留存每一个环节"：
        - 每个维度在 B1 被拒绝/通过的数量
        - A2 修正了多少个被拒绝的维度
        - B2 又拒绝了多少个
        - C 仲裁了多少个
        """
        total_dims = 0
        b1_rejected = 0
        a2_changed = 0       # A2实际改变了标签
        a2_fixed = 0         # A2把错的改对了（相对于金标准需在外部比对，这里只统计改变了）
        b2_rejected_after_a2 = 0
        c_used = 0
        c_changed = 0        # C改变了A2的标签

        final_sources = {"A1": 0, "A2": 0, "C": 0}
        per_dim_stats = {d: {"b1_rejected": 0, "a2_changed": 0, "b2_rejected": 0, "c_used": 0, "final_source": {"A1": 0, "A2": 0, "C": 0}} for d in Config.DIM_MAP}

        for res in trace_results:
            dim_trace = res.get("DimTrace", {})
            dim_changes = res.get("DimChanges", {})

            for d in Config.DIM_MAP:
                total_dims += 1
                dt = dim_trace.get(d, {})

                # B1 拒绝？
                if dt.get("B1_approved") is False:
                    b1_rejected += 1
                    per_dim_stats[d]["b1_rejected"] += 1

                # A2 改变了标签？
                a2c = dim_changes.get("A1_to_A2", {})
                if d in a2c:
                    a2_changed += 1
                    per_dim_stats[d]["a2_changed"] += 1

                # B2 拒绝？
                if dt.get("B2_approved") is False:
                    b2_rejected_after_a2 += 1
                    per_dim_stats[d]["b2_rejected"] += 1

                # C 使用？
                cc = dim_changes.get("A2_to_C", {})
                if d in cc:
                    c_used += 1
                    c_changed += 1
                    per_dim_stats[d]["c_used"] += 1

                # 最终来源
                fs = dt.get("final_source", "A1")
                if fs in final_sources:
                    final_sources[fs] += 1
                if fs in per_dim_stats[d]["final_source"]:
                    per_dim_stats[d]["final_source"][fs] += 1

        total = max(total_dims, 1)
        return {
            "total_dimensions": total_dims,
            "b1_rejected_dims": b1_rejected,
            "b1_pass_rate": (total_dims - b1_rejected) / total,
            "a2_actually_changed": a2_changed,
            "b2_rejected_after_a2": b2_rejected_after_a2,
            "c_used_dims": c_used,
            "c_changed_dims": c_changed,
            "final_source_distribution": final_sources,
            "per_dimension": per_dim_stats
        }

    def compute_pipeline_contribution(self, trace_results: List[Dict], df_gold: pd.DataFrame) -> Dict:
        """
        计算管道各阶段的贡献：
        - A1 alone 的 F1
        - A1 + B1(通过) + A2(修正) 的 F1
        - 完整 ABABC 的 F1
        - 每个阶段净提升了多少
        """
        def _calc_f1_from_source(trace_results, df_gold, source_fn):
            """通用：根据 source_fn(res, dim) 决定使用哪个标签"""
            y_true, y_pred = [], []
            for res in trace_results:
                row_id = str(res["ID"])
                gold_row = df_gold[df_gold["ID"].astype(str) == row_id]
                if gold_row.empty:
                    continue
                gold_row = gold_row.iloc[0]
                for dim_full, dim_pre in Config.DIM_MAP.items():
                    g_label = int(gold_row.get(f"{dim_pre}_label", 0))
                    p_label = source_fn(res, dim_full)
                    y_true.append(g_label)
                    y_pred.append(p_label)
            if not y_true:
                return 0.0
            return f1_score(y_true, y_pred, zero_division=0)

        # Stage 1: 只用 A1
        def use_a1(res, dim):
            return _get_dim_label(res.get("Steps", {}).get("A1", {}), dim)
        f1_a1 = _calc_f1_from_source(trace_results, df_gold, use_a1)

        # Stage 2: 用维度级AB（A1 + B1通过的维度用A1，B1拒绝的维度用A2）
        def use_dim_ab(res, dim):
            dt = res.get("DimTrace", {}).get(dim, {})
            if dt.get("B1_approved") is True:
                return _get_dim_label(res.get("Steps", {}).get("A1", {}), dim)
            else:
                return _get_dim_label(res.get("Steps", {}).get("A2", {}), dim)
        f1_ab = _calc_f1_from_source(trace_results, df_gold, use_dim_ab)

        # Stage 3: 完整 ABABC（final_label）
        def use_final(res, dim):
            val = res.get("DimTrace", {}).get(dim, {}).get("final_label")
            return val if val in (0, 1) else 0
        f1_final = _calc_f1_from_source(trace_results, df_gold, use_final)

        return {
            "f1_A1_only": f1_a1,
            "f1_A1_B1_A2": f1_ab,
            "f1_ABABC_full": f1_final,
            "improvement_A2_over_A1": f1_ab - f1_a1,
            "improvement_C_over_A2": f1_final - f1_ab,
            "total_improvement": f1_final - f1_a1
        }

    # ========== 旧接口（保留兼容） ==========

    def compute_role_interaction_stats(self, trace_results: List[Dict]) -> Dict:
        """样本级交互统计（兼容旧版，但统计更精确）"""
        stats = {
            "total_samples": len(trace_results),
            "exit_stage_counts": {},
            "a1_dim_pass_rate": 0.0,
            "revision_trigger_rate": 0.0,
            "revision_dim_fix_rate": 0.0,
            "deadlock_dim_rate": 0.0
        }

        # 用维度级数据计算
        dim_trans = self.compute_dimension_transitions(trace_results)
        stats["exit_stage_counts"] = {
            "Consensus_R1": sum(1 for r in trace_results if r.get("Exit_Stage") == "Consensus_R1"),
            "Consensus_R2": sum(1 for r in trace_results if r.get("Exit_Stage") == "Consensus_R2"),
            "Arbitrated": sum(1 for r in trace_results if "Arbitrat" in str(r.get("Exit_Stage", ""))),
            "Error": sum(1 for r in trace_results if "Error" in str(r.get("Exit_Stage", "")))
        }
        stats["a1_dim_pass_rate"] = dim_trans["b1_pass_rate"]
        stats["revision_trigger_rate"] = dim_trans["b1_rejected_dims"] / max(dim_trans["total_dimensions"], 1)
        stats["revision_dim_fix_rate"] = dim_trans["a2_actually_changed"] / max(dim_trans["b1_rejected_dims"], 1)
        stats["deadlock_dim_rate"] = dim_trans["c_used_dims"] / max(dim_trans["total_dimensions"], 1)

        return stats

    def compute_metrics_for_role(self, data_by_stage: Dict, role: str) -> Dict:
        """计算角色专属指标（同前）"""
        if role == "Annotator":
            y_true = data_by_stage["A1"]["y_true"]
            y_pred = data_by_stage["A1"]["y_pred"]
            reasons_pairs = data_by_stage["A1"]["reasons"]
        elif role == "Reviewer":
            y_true = data_by_stage["Final"]["y_true"]
            y_pred = data_by_stage["Final"]["y_pred"]
            reasons_pairs = []
        else:
            y_true = data_by_stage["Final"]["y_true"]
            y_pred = data_by_stage["Final"]["y_pred"]
            reasons_pairs = data_by_stage["Final"]["reasons"]

        if not y_true:
            return {
                "Recall": 0.0, "Precision": 0.0, "F1": 0.0,
                "RVS": 0.0, "Kappa": 0.0, "Composite_Score": 0.0, "Sample_Count": 0
            }

        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)

        rvs_scores = []
        for pred_reason, gold_reason in reasons_pairs:
            rvs_scores.append(self.calculate_rvs(pred_reason, gold_reason))
        rvs = np.mean(rvs_scores) if rvs_scores else 0.0

        weights = Config.ROLE_WEIGHTS.get(role, Config.ROLE_WEIGHTS["Annotator"])
        composite = (
            recall * weights["recall"] +
            precision * weights["precision"] +
            rvs * weights["rvs"] +
            kappa * weights["kappa"]
        )

        return {
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "RVS": rvs,
            "Kappa": kappa,
            "Composite_Score": composite,
            "Sample_Count": len(y_true)
        }

    def get_role_specific_bad_cases(self, trace_results: List[Dict], df_gold: pd.DataFrame, role: str, max_cases: int = 15) -> str:
        """按维度分组的错误案例收集，附带错误统计"""
        from collections import defaultdict

        # 按维度+错误类型分组收集
        dim_errors = defaultdict(lambda: {"FN": [], "FP": [], "LowRVS": [], "FalseReject": [], "MissedReject": [], "C_Wrong": []})
        dim_error_counts = defaultdict(lambda: {"FN": 0, "FP": 0, "LowRVS": 0, "FalseReject": 0, "MissedReject": 0, "C_Wrong": 0})

        for res in trace_results:
            row_id = str(res['ID'])
            gold_rows = df_gold[df_gold['ID'].astype(str) == row_id]
            if gold_rows.empty:
                continue
            gold_row = gold_rows.iloc[0]
            steps = res.get("Steps", {})
            dim_trace = res.get("DimTrace", {})
            text_snippet = str(gold_row.get('text', ''))[:80]

            for dim_full in Config.DIM_MAP:
                g_label = int(gold_row.get(f"{Config.DIM_MAP[dim_full]}_label", 0))
                g_reason = str(gold_row.get(f"{Config.DIM_MAP[dim_full]}_reasoning", ""))

                if role == "Annotator":
                    a1_output = steps.get("A1", {})
                    a1_label = _get_dim_label(a1_output, dim_full)
                    a1_reason = _get_dim_reasoning(a1_output, dim_full)
                    a1_conf = _get_dim_confidence(a1_output, dim_full)

                    if g_label == 1 and a1_label == 0:
                        dim_error_counts[dim_full]["FN"] += 1
                        if len(dim_errors[dim_full]["FN"]) < 2:
                            dim_errors[dim_full]["FN"].append(
                                f"[{text_snippet}] Pred=0 Gold=1 Conf={a1_conf} | {a1_reason[:80]}")
                    elif g_label == 0 and a1_label == 1:
                        dim_error_counts[dim_full]["FP"] += 1
                        if len(dim_errors[dim_full]["FP"]) < 2:
                            dim_errors[dim_full]["FP"].append(
                                f"[{text_snippet}] Pred=1 Gold=0 Conf={a1_conf} | {a1_reason[:80]}")
                    elif g_label == 1 and a1_label == 1:
                        rvs = self.calculate_rvs(a1_reason, g_reason) if a1_reason and g_reason else 0
                        if rvs < 0.7 and rvs > 0:
                            dim_error_counts[dim_full]["LowRVS"] += 1

                elif role == "Reviewer":
                    dt = dim_trace.get(dim_full, {})
                    b1_approved = dt.get("B1_approved")
                    b1_feedback = dt.get("B1_feedback", "")
                    a1_label = _get_dim_label(steps.get("A1", {}), dim_full)

                    if g_label == 0 and a1_label == 1 and b1_approved is not False:
                        dim_error_counts[dim_full]["MissedReject"] += 1
                        if len(dim_errors[dim_full]["MissedReject"]) < 2:
                            dim_errors[dim_full]["MissedReject"].append(
                                f"[{text_snippet}] FP approved by B1 | {b1_feedback[:60]}")
                    elif g_label == 1 and a1_label == 1 and b1_approved == False:
                        dim_error_counts[dim_full]["FalseReject"] += 1
                        if len(dim_errors[dim_full]["FalseReject"]) < 2:
                            dim_errors[dim_full]["FalseReject"].append(
                                f"[{text_snippet}] TP rejected by B1 | {b1_feedback[:60]}")

                else:  # Arbitrator
                    final_label = dim_trace.get(dim_full, {}).get("final_label", 0)
                    final_source = dim_trace.get(dim_full, {}).get("final_source", "")

                    if final_source == "C" and g_label != final_label:
                        dim_error_counts[dim_full]["C_Wrong"] += 1
                        if len(dim_errors[dim_full]["C_Wrong"]) < 2:
                            dim_errors[dim_full]["C_Wrong"].append(
                                f"[{text_snippet}] C裁决错误 Gold={g_label} Pred={final_label}")

        # 生成错误统计 + 代表性案例
        error_types_map = {
            "FN": ("Missed Signal", "❌"), "FP": ("Hallucination", "❌"),
            "LowRVS": ("Low RVS", "⚠️"), "FalseReject": ("False Rejection", "❌"),
            "MissedReject": ("Missed Reject", "❌"), "C_Wrong": ("C Wrong", "❌")
        }

        lines = []
        total_errors = 0
        # 先输出统计摘要
        for dim_full in Config.DIM_MAP:
            dim_total = sum(dim_error_counts[dim_full].values())
            if dim_total == 0:
                continue
            total_errors += dim_total
            dim_short = dim_full.split(" ")[0] if " " in dim_full else dim_full[:15]
            parts = []
            for ek, (elabel, _) in error_types_map.items():
                cnt = dim_error_counts[dim_full][ek]
                if cnt > 0:
                    parts.append(f"{elabel}={cnt}")
            lines.append(f"  {dim_short}: {', '.join(parts)}")

        if total_errors == 0:
            return "✅ No significant errors detected in this round."

        summary = f"\n=== ERROR PATTERNS (Total: {total_errors} errors across dimensions) ===\n" + "\n".join(lines)

        # 再输出代表性案例（每种错误类型最多2例，优先高频维度）
        examples = []
        example_count = 0
        for dim_full in Config.DIM_MAP:
            if example_count >= max_cases:
                break
            for ek in ["FN", "FP", "FalseReject", "MissedReject", "C_Wrong"]:
                for case in dim_errors[dim_full][ek][:2]:
                    if example_count >= max_cases:
                        break
                    examples.append(f"[{dim_full}] {case}")
                    example_count += 1

        if examples:
            summary += f"\n\n=== REPRESENTATIVE ERROR CASES ({len(examples)} examples) ===\n"
            summary += "\n".join(examples)

        return summary

    def compute_all_metrics(self, trace_results: List[Dict], df_gold: pd.DataFrame) -> Tuple:
        """
        主入口：计算各项指标
        返回: (system_metrics, role_metrics, interaction_stats, dim_transitions, pipeline_contribution, detailed_data)
        """
        system_metrics = self.compute_system_final_metrics(trace_results, df_gold)
        data_by_stage = self.extract_labels_and_reasons(trace_results, df_gold)

        role_metrics = {}
        for role in ["Annotator", "Reviewer", "Arbitrator"]:
            role_metrics[role] = self.compute_metrics_for_role(data_by_stage, role)

        interaction_stats = self.compute_role_interaction_stats(trace_results)
        dim_transitions = self.compute_dimension_transitions(trace_results)
        pipeline_contribution = self.compute_pipeline_contribution(trace_results, df_gold)

        return system_metrics, role_metrics, interaction_stats, dim_transitions, pipeline_contribution, data_by_stage


# ===================== 6. 维度级跟踪工具 =====================

def _get_dim_keywords(obj, dim_name):
    """从维度对象中提取关键词"""
    if not isinstance(obj, dict):
        return []
    dim_data = obj.get(dim_name, {})
    if not isinstance(dim_data, dict):
        return []
    kws = dim_data.get("keywords", [])
    return kws if isinstance(kws, list) else []





# ===================== 7. 业务逻辑层 (维度级Trace Pipeline) =====================

async def run_trace_pipeline(manager, row_id, text, prompts, lexicon=""):
    """
    维度级 A-B-A-B-C 闭环：
    - 6 个维度各自走 A1→B1(通过/拒绝)→A2(仅拒绝的维度)→B2→C(仅仍有争议的维度)
    - 完整留存每个维度在每个阶段的标签、置信度、推理
    - 记录每个维度的最终来源 (A1/A2/C)
    """
    trace = {
        "ID": str(row_id),
        "Steps": {},          # A1/A2/C的原始完整输出（保留向后兼容）
        "DimTrace": _build_dim_trace_empty(list(Config.DIM_MAP.keys())),  # 维度级跟踪
        "DimChanges": {},     # 各阶段的维度变化记录
        "Final_Output": {},
        "Exit_Stage": "Unknown"
    }

    sys_a = prompts['Annotator']
    sys_b = prompts['Reviewer']
    sys_c = prompts['Arbitrator']

    # ============ A1: 首次标注（全6维） ============
    msg_a1 = [
        {"role": "system", "content": sys_a},
        {"role": "user", "content": f"Text: {text}\nLexicon: {lexicon}"}
    ]
    res_a1_raw = await manager.call(Config.WORKER_MODELS['Annotator'], msg_a1)
    trace["Steps"]["A1"] = manager._safe_json_parse(res_a1_raw, "Annotator_A1")

    if "error" in trace["Steps"]["A1"]:
        trace["Final_Output"] = {"error": "A1_failed"}
        trace["Exit_Stage"] = "Error_A1"
        return trace

    # 将A1结果写入DimTrace
    for dim_name in Config.DIM_MAP:
        trace["DimTrace"][dim_name]["A1"] = trace["Steps"]["A1"].get(dim_name, {})
        trace["DimTrace"][dim_name]["final_label"] = _get_dim_label(trace["Steps"]["A1"], dim_name)
        trace["DimTrace"][dim_name]["final_source"] = "A1"
        trace["DimTrace"][dim_name]["final_confidence"] = _get_dim_confidence(trace["Steps"]["A1"], dim_name)

    # ============ B1: 首次审核（维度级） ============
    inp_b1 = json.dumps({"text": text, "annotation": trace["Steps"]["A1"]}, ensure_ascii=False)
    msg_b1 = [
        {"role": "system", "content": sys_b},
        {"role": "user", "content": inp_b1}
    ]
    res_b1_raw = await manager.call(Config.WORKER_MODELS['Reviewer'], msg_b1)
    trace["Steps"]["B1"] = manager._safe_json_parse(res_b1_raw, "Reviewer_B1")

    if "error" in trace["Steps"]["B1"]:
        # B1失败：保守策略，接受A1
        trace["Final_Output"] = trace["Steps"]["A1"]
        trace["Exit_Stage"] = "B1_Error_Fallback"
        return trace

    # 解析B1的维度级审核结论
    b1_verdicts = _parse_b_verdicts(trace["Steps"]["B1"], list(Config.DIM_MAP.keys()))
    for dim_name in Config.DIM_MAP:
        v = b1_verdicts.get(dim_name, {"approved": True, "feedback": ""})
        trace["DimTrace"][dim_name]["B1_approved"] = v["approved"]
        trace["DimTrace"][dim_name]["B1_feedback"] = v["feedback"]

    # 统计B1拒绝了几个维度
    b1_rejected = [d for d in Config.DIM_MAP if not trace["DimTrace"][d]["B1_approved"]]

    # 如果B1批准了所有维度 → Consensus_R1（维度级）
    if not b1_rejected:
        trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], list(Config.DIM_MAP.keys()))
        trace["Exit_Stage"] = "Consensus_R1"
        return trace

    # ============ A2: 修订（仅B1拒绝的维度） ============
    # 传给A2：只要求重新标注被拒绝的维度，已通过的保留
    rejected_detail = {}
    for d in b1_rejected:
        rejected_detail[d] = {
            "A1_label": _get_dim_label(trace["Steps"]["A1"], d),
            "A1_reasoning": _get_dim_reasoning(trace["Steps"]["A1"], d),
            "B1_feedback": trace["DimTrace"][d]["B1_feedback"]
        }

    a2_instruction = (
        f"Text: {text}\n"
        f"Lexicon: {lexicon}\n\n"
        f"Below are the 6 CSM dimensions. "
        f"Dimensions that PASSED review: keep their current labels unchanged.\n"
        f"Dimensions that FAILED review (listed below): REVISE based on feedback.\n\n"
        f"REVISIONS NEEDED ({len(b1_rejected)} dimensions):\n"
        f"{json.dumps(rejected_detail, ensure_ascii=False, indent=2)}\n\n"
        f"Output ALL 6 dimensions. For passed dimensions, output the SAME label as before. "
        f"For failed dimensions, output the corrected label with improved reasoning."
    )
    msg_a2 = [
        {"role": "system", "content": sys_a},
        {"role": "user", "content": a2_instruction}
    ]
    res_a2_raw = await manager.call(Config.WORKER_MODELS['Annotator'], msg_a2)
    trace["Steps"]["A2"] = manager._safe_json_parse(res_a2_raw, "Annotator_A2")

    if "error" in trace["Steps"]["A2"]:
        # A2失败：接受A1（保守）
        trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], list(Config.DIM_MAP.keys()))
        trace["Exit_Stage"] = "A2_Error_Fallback"
        return trace

    # 比较A1→A2，检测A2实际改变了哪些维度
    a2_changed, a2_changes = _detect_changed_dimensions(trace["Steps"]["A1"], trace["Steps"]["A2"], list(Config.DIM_MAP.keys()))
    trace["DimChanges"]["A1_to_A2"] = a2_changes

    # 将A2结果写入DimTrace（仅覆盖被拒绝的维度）
    for dim_name in Config.DIM_MAP:
        trace["DimTrace"][dim_name]["A2"] = trace["Steps"]["A2"].get(dim_name, {})

    # 关键：对B1拒绝且A2改变了标签的维度 → 更新final_label
    # 对B1拒绝但A2没改变标签的维度 → 保持A1（A2没有修正）
    for d in b1_rejected:
        if d in a2_changed:
            trace["DimTrace"][d]["final_label"] = _get_dim_label(trace["Steps"]["A2"], d)
            trace["DimTrace"][d]["final_source"] = "A2"
            trace["DimTrace"][d]["final_confidence"] = _get_dim_confidence(trace["Steps"]["A2"], d)

    # ============ B2: 二次审核（仅检查A2实际改变了的维度） ============
    # 如果没有维度被改变 → 不同B2，直接走C或接受
    if not a2_changed:
        # A2没有任何改变 → A2没有修正问题，跳过B2直接给C仲裁
        pass  # 继续到C
    else:
        # 构造B2的检查重点：只核查A2改变了的维度
        b2_focus = {d: {
            "A1_label": trace["DimChanges"]["A1_to_A2"][d]["from"],
            "A2_new_label": trace["DimChanges"]["A1_to_A2"][d]["to"],
            "A2_reasoning": _get_dim_reasoning(trace["Steps"]["A2"], d),
            "B1_feedback": trace["DimTrace"][d]["B1_feedback"]
        } for d in a2_changed}

        b2_instruction = (
            f"Text: {text}\n\n"
            f"The Annotator has revised {len(a2_changed)} dimension(s). "
            f"Review ONLY these revised dimensions for correctness.\n"
            f"Dimensions not listed below are unchanged and do not need re-review.\n\n"
            f"REVISED DIMENSIONS:\n"
            f"{json.dumps(b2_focus, ensure_ascii=False, indent=2)}\n\n"
            f"For each revised dimension, determine if the NEW label is correct. "
            f"Output JSON with 'dimension_feedback' per dimension indicating pass/fail."
        )
        msg_b2 = [
            {"role": "system", "content": sys_b},
            {"role": "user", "content": b2_instruction}
        ]
        res_b2_raw = await manager.call(Config.WORKER_MODELS['Reviewer'], msg_b2)
        trace["Steps"]["B2"] = manager._safe_json_parse(res_b2_raw, "Reviewer_B2")

        if "error" not in trace["Steps"]["B2"]:
            b2_verdicts = _parse_b_verdicts(trace["Steps"]["B2"], list(Config.DIM_MAP.keys()))
            for dim_name in Config.DIM_MAP:
                v = b2_verdicts.get(dim_name, {"approved": True, "feedback": ""})
                trace["DimTrace"][dim_name]["B2_approved"] = v["approved"]
                trace["DimTrace"][dim_name]["B2_feedback"] = v["feedback"]

    # 计算B2后仍有几个维度被拒绝（仅在B2实际执行后）
    b2_rejected = []  # 默认空列表，防止 a2_changed 为空时引用未定义变量
    if a2_changed:
        b2_rejected = [
            d for d in Config.DIM_MAP
            if trace["DimTrace"][d]["B2_approved"] is False
        ]

    # 如果A2没有改变任何维度 → 直接送C
    # 如果B2批准了所有被A2改变的维度 → Consensus_R2
    if a2_changed and not b2_rejected:
        trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], list(Config.DIM_MAP.keys()))
        trace["Exit_Stage"] = "Consensus_R2"
        return trace

    # ============ C: 仲裁（仅仍有争议的维度） ============
    # 收集需要仲裁的维度信息
    c_disputed = {}
    if a2_changed:
        # 有A2修改且B2拒绝了某些 → 仲裁那些维度
        dispute_dims = b2_rejected if b2_rejected else a2_changed
    else:
        # A2没有修改 → 仲裁B1拒绝但A2没能修正的维度
        dispute_dims = b1_rejected

    for d in dispute_dims:
        c_disputed[d] = {
            "A1": {
                "label": _get_dim_label(trace["Steps"]["A1"], d),
                "reasoning": _get_dim_reasoning(trace["Steps"]["A1"], d),
                "keywords": _get_dim_keywords(trace["Steps"]["A1"], d)
            },
            "A2": {
                "label": _get_dim_label(trace["Steps"]["A2"], d),
                "reasoning": _get_dim_reasoning(trace["Steps"]["A2"], d),
                "keywords": _get_dim_keywords(trace["Steps"]["A2"], d)
            },
            "B1_feedback": trace["DimTrace"][d]["B1_feedback"],
            "B2_feedback": trace["DimTrace"][d]["B2_feedback"]
        }

    c_instruction = (
        f"Text: {text}\n\n"
        f"The Annotator and Reviewer disagree on {len(dispute_dims)} dimension(s). "
        f"Make a FINAL decision for each disputed dimension.\n\n"
        f"DISPUTED DIMENSIONS:\n"
        f"{json.dumps(c_disputed, ensure_ascii=False, indent=2)}\n\n"
        f"For each disputed dimension, output your final label, reasoning, and confidence. "
        f"For dimensions NOT listed, output them unchanged from A2's annotation."
    )
    msg_c = [
        {"role": "system", "content": sys_c},
        {"role": "user", "content": c_instruction}
    ]
    res_c_raw = await manager.call(Config.WORKER_MODELS['Arbitrator'], msg_c)
    trace["Steps"]["C"] = manager._safe_json_parse(res_c_raw, "Arbitrator_C")

    if "error" not in trace["Steps"]["C"] and isinstance(trace["Steps"]["C"], dict):
        # 检测C改变了A2的哪些维度
        c_changed, c_changes = _detect_changed_dimensions(trace["Steps"]["A2"], trace["Steps"]["C"], list(Config.DIM_MAP.keys()))
        trace["DimChanges"]["A2_to_C"] = c_changes

        for dim_name in Config.DIM_MAP:
            trace["DimTrace"][dim_name]["C"] = trace["Steps"]["C"].get(dim_name, {})

        # 对争议维度：用C的仲裁结果
        for d in dispute_dims:
            if d in c_changed:
                trace["DimTrace"][d]["final_label"] = _get_dim_label(trace["Steps"]["C"], d)
                trace["DimTrace"][d]["final_source"] = "C"
                trace["DimTrace"][d]["final_confidence"] = _get_dim_confidence(trace["Steps"]["C"], d)
    else:
        # C失败：保持当前最优状态
        trace["Exit_Stage"] = "C_Error_Fallback"
        trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], list(Config.DIM_MAP.keys()))
        return trace

    # ============ 构建最终输出并汇总状态 ============
    trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], list(Config.DIM_MAP.keys()))

    # 汇总退出状态: 每个维度经历了什么阶段
    stages_used = set()
    for d in Config.DIM_MAP:
        stages_used.add(trace["DimTrace"][d]["final_source"])
    if stages_used == {"A1"}:
        trace["Exit_Stage"] = "Consensus_R1"
    elif stages_used == {"A1", "A2"} or stages_used == {"A2"}:
        trace["Exit_Stage"] = "Consensus_R2"
    else:
        trace["Exit_Stage"] = "Arbitrated"

    # 添加汇总统计
    trace["DimSummary"] = {
        d: {
            "final_label": trace["DimTrace"][d]["final_label"],
            "final_source": trace["DimTrace"][d]["final_source"],
            "b1_approved": trace["DimTrace"][d]["B1_approved"],
            "b2_approved": trace["DimTrace"][d]["B2_approved"],
            "final_confidence": trace["DimTrace"][d]["final_confidence"]
        }
        for d in Config.DIM_MAP
    }

    return trace


# ===================== 7. 辅助函数 =====================

def extract_prompt_from_xml(raw_response):
    """从XML响应中提取提示词"""
    if not isinstance(raw_response, str):
        return None

    pattern = r'<H_RAMOS_PROMPT_V1>(.*?)</H_RAMOS_PROMPT_V1>'
    match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)

    if match:
        content = match.group(1).strip()
        content = re.sub(r'^```\w*\s*|\s*```$', '', content)
        return content

    return None


def ensure_lexicon_placeholder(prompt_text, role="Annotator"):
    """确保 {lexicon} 占位符存在"""
    if role != "Annotator":
        return prompt_text

    if "{lexicon}" in prompt_text:
        return prompt_text

    print(f"    🚨 CRITICAL: {{lexicon}} placeholder missing! Attempting repair...")

    variants = ['{ lexicon }', '{Lexicon}', '{LEXICON}', '{{lexicon}}', '[lexicon]', '(lexicon)']
    for variant in variants:
        if variant in prompt_text:
            prompt_text = prompt_text.replace(variant, "{lexicon}")
            print(f"    🔧 Fixed: Replaced '{variant}' with '{{lexicon}}'")
            return prompt_text

    insertion_points = [
        "# 🔄 Protocol: Input Analysis",
        "# 📥 Input Format",
        "# Input Format",
        "Your Input will be formatted as:"
    ]

    for point in insertion_points:
        if point in prompt_text:
            idx = prompt_text.find(point)
            next_section = prompt_text.find("\n#", idx + 1)
            if next_section == -1:
                next_section = len(prompt_text)

            insertion = f"\n\n**IMPORTANT**: The dynamic lexicon will be provided via the `{{lexicon}}` placeholder. Ensure your prompt references it.\n"
            prompt_text = prompt_text[:next_section] + insertion + prompt_text[next_section:]
            print(f"    🔧 Fixed: Inserted '{{lexicon}}' reference near '{point}'")
            return prompt_text

    prompt_text += "\n\n# [AUTO-FIXED] Dynamic Lexicon Reference\nThe lexicon data is passed via the {lexicon} placeholder in the input.\n"
    print(f"    🔧 Fixed: Appended '{{lexicon}}' reference to end of prompt")
    return prompt_text


def clean_and_save_prompt(raw_content, role, round_i):
    """整合XML解析和 {lexicon} 保护"""
    if not raw_content:
        print(f"    ⚠️ Empty content for {role}, skipping save")
        return None

    prompt_text = extract_prompt_from_xml(raw_content)

    if prompt_text is None:
        if isinstance(raw_content, dict):
            prompt_text = raw_content.get("system_prompt_template", str(raw_content))
        else:
            prompt_text = str(raw_content)
        print(f"    ⚠️ XML extraction failed for {role}, using raw content")

    if role == "Annotator":
        prompt_text = ensure_lexicon_placeholder(prompt_text, role)

        if "{lexicon}" not in prompt_text:
            print(f"    ❌ FAILED: Could not repair {{lexicon}} placeholder for {role}")
            return None

    if len(prompt_text) < 100:
        print(f"    ⚠️ Warning: {role} prompt suspiciously short ({len(prompt_text)} chars)")

    required_sections = ["# System Role", "Output Schema"]
    missing = [s for s in required_sections if s not in prompt_text]
    if missing:
        print(f"    ⚠️ Warning: {role} prompt missing sections: {missing}")

    filepath = os.path.join(Config.HISTORY_ROOT, f"{role}_v{round_i + 1}.txt")
    with open(filepath, "w", encoding='utf-8') as f:
        f.write(prompt_text)

    print(f"    ✅ {role}_v{round_i + 1}.txt saved ({len(prompt_text)} chars, {'{lexicon} OK' if role == 'Annotator' else 'N/A'})")
    return prompt_text


def load_initial_prompts(history_dir: str, baseline_dir: str) -> dict:
    """
    智能加载提示词系统：
    1. 优先检查 'prompts_history' 是否有断点存档 (v1, v2...)。
    2. 如果没有存档，强制加载 'prompts/2_pipeline_roles' 下的 v0.json 基线文件。
    3. 如果文件缺失，才降级使用代码内置的 DEFAULT_PROMPTS。
    """
    prompts = {}
    roles = ["Annotator", "Reviewer", "Arbitrator"]

    print(f"\n🔍 正在初始化提示词加载系统...")

    # =========================================================
    # 1. 尝试断点续传 (检查 history 目录)
    # =========================================================
    if os.path.exists(history_dir) and os.listdir(history_dir):
        print(f"   📂 检测到历史存档目录: {history_dir}")
        latest_prompts = {}
        for role in roles:
            # 找类似 Annotator_v5.txt 的文件
            files = [f for f in os.listdir(history_dir) if f.startswith(role) and f.endswith(".txt")]
            if files:
                # 提取版本号最大的
                files.sort(key=lambda x: int(re.search(r'_v(\d+)', x).group(1)) if re.search(r'_v(\d+)', x) else 0)
                latest_file = files[-1]
                try:
                    with open(os.path.join(history_dir, latest_file), 'r', encoding='utf-8') as f:
                        content = f.read()
                    # XML 清洗
                    text = extract_prompt_from_xml(content) or content
                    latest_prompts[role] = text
                    print(f"   RESUME: 已恢复 {role} 至版本 {latest_file}")
                except Exception as e:
                    print(f"   ⚠️ 恢复 {role} 失败: {e}")

        # 如果成功加载了历史版本，直接返回，不再读取基线
        if len(latest_prompts) == 3:
            return latest_prompts
        else:
            prompts.update(latest_prompts)

    # =========================================================
    # 2. 加载用户自定义基线 (v0.json) - 这是您最需要的部分！
    # =========================================================
    if not prompts and os.path.exists(baseline_dir):
        print(f"   📂 正在加载自定义基线: {baseline_dir}")

        # 定义文件名映射 (根据您提供的文件名)
        file_map = {
            "Annotator": "annotator_v0.json",
            "Reviewer": "reviewer_v0.json",
            "Arbitrator": "arbitrator_v0.json"
        }

        for role, filename in file_map.items():
            filepath = os.path.join(baseline_dir, filename)

            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # ⚠️ JSON 解析逻辑：根据您的 JSON 结构提取 Prompt 文本
                    # 通常 Prompt 存在于 "content", "system_prompt", "prompt" 或 "text" 字段中
                    # 如果找不到这些 Key，就假设整个 JSON 的字符串表示就是 Prompt (兜底)
                    prompt_text = ""
                    if isinstance(data, dict):
                        # 尝试常见的 Key
                        for key in ["system_prompt", "content", "prompt", "text", "template"]:
                            if key in data and isinstance(data[key], str):
                                prompt_text = data[key]
                                break
                        # 如果还没找到，且 JSON 很小，可能 Value 就是 Prompt
                        if not prompt_text:
                            # 找最长的那个字符串 Value
                            str_values = [v for v in data.values() if isinstance(v, str)]
                            if str_values:
                                prompt_text = max(str_values, key=len)
                    elif isinstance(data, str):
                        prompt_text = data

                    # 最后的兜底：如果解析失败，转成字符串
                    if not prompt_text:
                        prompt_text = json.dumps(data, ensure_ascii=False, indent=2)
                        print(f"   ⚠️ 警告: 未能识别 JSON 结构，已将整个 JSON 内容作为 Prompt。")

                    # 注入 Lexicon 占位符 (针对 Annotator)
                    if role == "Annotator":
                        prompt_text = ensure_lexicon_placeholder(prompt_text, role)

                    prompts[role] = prompt_text
                    print(f"   ✅ BASELINE: 成功加载 {filename}")

                except Exception as e:
                    print(f"   ❌ 加载 {filename} 失败: {e}")
            else:
                print(f"   ⚠️ 文件缺失: {filename}")

    # =========================================================
    # 3. 最后的兜底 (使用代码内置默认值)
    # =========================================================
    for role in roles:
        if role not in prompts:
            print(f"   ℹ️ DEFAULT: {role} 使用内置默认提示词 (未找到文件)")
            prompts[role] = PromptLibrary.DEFAULT_PROMPTS[role]

            # 确保内置的 Annotator 也有 lexicon 占位符
            if role == "Annotator":
                prompts[role] = ensure_lexicon_placeholder(prompts[role], role)

    return prompts


def save_best_prompts(prompts: dict, round_i: int, score: float):
    """保存最佳提示词到专用目录"""
    os.makedirs(Config.BEST_PROMPT_DIR, exist_ok=True)

    for role in ["Annotator", "Reviewer", "Arbitrator"]:
        src = os.path.join(Config.HISTORY_ROOT, f"{role}_v{round_i + 1}.txt")
        dst = os.path.join(Config.BEST_PROMPT_DIR, f"{role}_BEST_R{round_i}.txt")

        if os.path.exists(src):
            shutil.copy(src, dst)

    # 保存元数据
    meta_path = os.path.join(Config.BEST_PROMPT_DIR, "best_metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            "round": round_i,
            "kappa": score,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, ensure_ascii=False, indent=2)

    print(f"   💾 Best prompts saved to {Config.BEST_PROMPT_DIR}")


# ===================== 8. 主程序 (System-Final Optimization) =====================

RUN_MODE = "full"
DEBUG_MODE = False


# ✅ 全局状态变量（用于早停）
optimization_state = {
    "best_score": 0.0,
    "best_round": 0,
    "best_prompts": None,
    "patience_counter": 0,
    "stopped": False
}


async def main():
    print("🚀 H-RAMOS Research Edition - System Final Optimization")
    print(f"⚙️ Run Mode: {RUN_MODE} | Debug: {DEBUG_MODE}")
    print(f"🎯 Target: Kappa ≥ {Config.STOPPING_CRITERIA['target_kappa']}, "
          f"F1 ≥ {Config.STOPPING_CRITERIA['target_f1']}")
    print("🔧 Features: XML Protocol + Lexicon Protection + Bilingual Format + Early Stopping")

    # 创建目录
    for path in [Config.HISTORY_ROOT, Config.LOG_ROOT, Config.TRACE_ROOT, Config.BEST_PROMPT_DIR]:
        os.makedirs(path, exist_ok=True)



    # 初始化词典管理器
    lexicon_manager = LexiconManager(Config.LEXICON_FILE)
    init_stats = lexicon_manager.get_stats()
    print(f"\n📚 Initial Lexicon: {sum(init_stats.values())} words")
    for dim, count in init_stats.items():
        print(f"   {dim}: {count} words")

    # 加载提示词
    current_prompts = PromptLibrary.DEFAULT_PROMPTS.copy()
    current_prompts = load_initial_prompts(Config.HISTORY_ROOT, Config.BASELINE_PROMPT_DIR)
    # 注入 Lexicon (确保 Annotator 拿到最新的词典)
    if "{lexicon}" in current_prompts["Annotator"]:
        current_prompts["Annotator"] = current_prompts["Annotator"].replace(
            "{lexicon}", lexicon_manager.get_lexicon_for_prompt()
        )
        print("   📖 Injected lexicon into Annotator prompt")
    # 初始化管理器与评估器
    manager = AsyncLLMManager()
    evaluator = Evaluator()
    memory = OptimizationMemory()

    try:
        # 加载测试数据
        if os.path.exists(Config.INPUT_FILE):
            df_test = pd.read_excel(Config.INPUT_FILE)
            df_test['ID'] = df_test['ID'].astype(str)
            print(f"\n📊 Loaded {len(df_test)} test samples")
        else:
            print("❌ Data file missing.")
            return

        # ================================================================
        # BASELINE ROUNDS (审稿人Comment 32回复)
        # 增加两个基线来证明: (1) 专家提示词设计是主要贡献 (2) 优化提供增量改进
        # Round 0a: 零提示词基线 (无system prompt)
        # Round 0b: 单句提示词基线 (one-shot minimal instruction)
        # Round 0c: 专家v0提示词 (expert-designed, 优化起点)
        # ================================================================

        baseline_results = []

        # --- Baseline A: Zero-prompt (no system prompt, just user text) ---
        print(f"\n{'='*70}")
        print(f"BASELINE A: Zero-Prompt (No system instruction)")
        print(f"{'='*70}")
        zero_prompts = {
            "Annotator": "You are a helpful assistant. Output valid JSON.",
            "Reviewer": "You are a helpful assistant. Output valid JSON.",
            "Arbitrator": "You are a helpful assistant. Output valid JSON."
        }
        tasks_a = [run_trace_pipeline(manager, r['ID'], r['text'], zero_prompts) for _, r in df_test.iterrows()]
        trace_a = await tqdm.gather(*tasks_a, desc="Zero-prompt baseline")
        ResultLogger.save_trace(0, trace_a, suffix="_baseline_zero")
        sys_a, _, _, _, _, _ = evaluator.compute_all_metrics(trace_a, df_test)
        baseline_results.append({"round": "0a_zero_prompt", "kappa": sys_a['Kappa'], "f1": sys_a['F1']})
        print(f"   Zero-prompt Kappa={sys_a['Kappa']:.4f}, F1={sys_a['F1']:.4f}")

        # --- Baseline B: One-shot minimal instruction ---
        print(f"\n{'='*70}")
        print(f"BASELINE B: One-Shot Minimal Instruction")
        print(f"{'='*70}")
        one_shot_prompts = {
            "Annotator": "Classify this dental pain social media post into 6 CSM dimensions: Perceived Cause, Symptom Description, Perceived Consequences, Coping and Management, Emotional Expression, Social Interaction. For each dimension, output label (0/1) with confidence (2.0-5.0). Output valid JSON only.",
            "Reviewer": "Review the annotation for correctness. Output valid JSON with dimension_feedback.",
            "Arbitrator": "Resolve annotation disputes. Output final valid JSON."
        }
        tasks_b = [run_trace_pipeline(manager, r['ID'], r['text'], one_shot_prompts) for _, r in df_test.iterrows()]
        trace_b = await tqdm.gather(*tasks_b, desc="One-shot baseline")
        ResultLogger.save_trace(0, trace_b, suffix="_baseline_oneshot")
        sys_b, _, _, _, _, _ = evaluator.compute_all_metrics(trace_b, df_test)
        baseline_results.append({"round": "0b_one_shot", "kappa": sys_b['Kappa'], "f1": sys_b['F1']})
        print(f"   One-shot Kappa={sys_b['Kappa']:.4f}, F1={sys_b['F1']:.4f}")

        # Save baseline comparison
        import pandas as pd
        df_baseline = pd.DataFrame(baseline_results)
        baseline_path = os.path.join(Config.LOG_ROOT, "00_baseline_comparison.xlsx")
        df_baseline.to_excel(baseline_path, index=False)
        print(f"\n   Baseline comparison saved to {baseline_path}")
        print(f"   Improvement from zero-prompt to v0-expert: +{sys_a['Kappa']:.4f} Kappa ({sys_a['Kappa']:.4f} -> to be measured vs Round 1)")

        # 完整优化循环（带早停）
        for round_i in range(1, Config.STOPPING_CRITERIA['max_rounds'] + 1):
            if optimization_state["stopped"]:
                break

            print(f"\n{'=' * 70}")
            print(f"ROUND {round_i}/{Config.STOPPING_CRITERIA['max_rounds']} - Target Kappa: {Config.STOPPING_CRITERIA['target_kappa']}")
            print(f"{'=' * 70}")
            # === ✅ 新增：智能回滚机制 ===
            # 如果上一轮效果严重下滑（比如下降超过 1%），强制回滚到历史最佳提示词
            # 这样 Meta-LLM 就会基于"最好的版本"尝试新的优化方向，而不是在"烂版本"上修修补补
            if round_i > 1 and optimization_state["best_prompts"]:
                current_kappa = memory.system_history[-1]['kappa']
                best_kappa = optimization_state["best_score"]

                # 如果当前由于"负优化"导致比最佳值低了 0.01 以上
                if current_kappa < best_kappa - 0.01:
                    print(f"   ⚠️ 检测到性能衰退 (Current: {current_kappa:.4f} < Best: {best_kappa:.4f})")
                    print(f"   🔙 正在回滚至 Round {optimization_state['best_round']} 的最佳提示词...")
                    current_prompts = optimization_state["best_prompts"].copy()
            # ============================
            # 更新词典
            if "{lexicon}" in current_prompts["Annotator"]:
                current_prompts["Annotator"] = current_prompts["Annotator"].replace(
                    "{lexicon}", lexicon_manager.get_lexicon_for_prompt()
                )

            # Phase 1: 运行完整Pipeline
            tasks = [run_trace_pipeline(manager, r['ID'], r['text'], current_prompts) for _, r in df_test.iterrows()]
            trace_results = await tqdm.gather(*tasks, desc="Pipeline Running")

            # Phase 2: 保存日志
            ResultLogger.save_trace(round_i, trace_results)
            ResultLogger.save_trace_per_model(round_i, trace_results)

            # Phase 3: 评估（系统最终指标 + 维度级过渡统计）
            (system_metrics, role_metrics, interaction_stats,
             dim_transitions, pipeline_contrib, detailed_data) = evaluator.compute_all_metrics(trace_results, df_test)

            # ✅ 记录系统指标并检查早停
            system_delta = memory.record_system(round_i, system_metrics)

            # 打印系统最终质量报告（核心）
            print(f"\n{'='*70}")
            print(f"🎯 SYSTEM FINAL OUTPUT QUALITY (This is what we optimize!)")
            print(f"{'='*70}")
            print(f"Kappa:     {system_metrics['Kappa']:.4f} (Target: {Config.STOPPING_CRITERIA['target_kappa']}) {'✅' if system_metrics['Kappa'] >= Config.STOPPING_CRITERIA['target_kappa'] else '🔄'}")
            print(f"F1-Score:  {system_metrics['F1']:.4f} (Target: {Config.STOPPING_CRITERIA['target_f1']}) {'✅' if system_metrics['F1'] >= Config.STOPPING_CRITERIA['target_f1'] else '🔄'}")
            print(f"Recall:    {system_metrics['Recall']:.4f}")
            print(f"Precision: {system_metrics['Precision']:.4f}")
            print(f"Samples:   {system_metrics['Support']}")

            # 打印管道各阶段贡献
            print(f"\n📈 PIPELINE CONTRIBUTION (F1 by stage)")
            print(f"{'-'*70}")
            print(f"A1 alone:       {pipeline_contrib['f1_A1_only']:.4f}")
            print(f"A1+B1+A2:       {pipeline_contrib['f1_A1_B1_A2']:.4f}  (+{pipeline_contrib['improvement_A2_over_A1']:+.4f})")
            print(f"ABABC full:     {pipeline_contrib['f1_ABABC_full']:.4f}  (+{pipeline_contrib['improvement_C_over_A2']:+.4f})")
            print(f"Total improve:  {pipeline_contrib['total_improvement']:+.4f}")

            # 打印维度级过渡统计
            print(f"\n🔄 DIMENSION TRANSITIONS")
            print(f"{'-'*70}")
            print(f"Total dims:     {dim_transitions['total_dimensions']}")
            print(f"B1 rejected:    {dim_transitions['b1_rejected_dims']}  (pass rate: {dim_transitions['b1_pass_rate']:.1%})")
            print(f"A2 changed:     {dim_transitions['a2_actually_changed']}  (of those rejected by B1)")
            print(f"B2 re-rejected: {dim_transitions['b2_rejected_after_a2']}")
            print(f"C arbitrated:   {dim_transitions['c_used_dims']}  (changed {dim_transitions['c_changed_dims']})")
            print(f"Final source:   A1={dim_transitions['final_source_distribution'].get('A1',0)}  "
                  f"A2={dim_transitions['final_source_distribution'].get('A2',0)}  "
                  f"C={dim_transitions['final_source_distribution'].get('C',0)}")

            # 打印每维度详情
            print(f"\n📊 PER-DIMENSION BREAKDOWN")
            print(f"{'─'*70}")
            for d in Config.DIM_MAP:
                ps = dim_transitions['per_dimension'][d]
                b1r = ps['b1_rejected']
                a2c = ps['a2_changed']
                fs = ps['final_source']
                print(f"  {d:25} B1✗={b1r} A2Δ={a2c} "
                      f"Final→A1:{fs['A1']} A2:{fs['A2']} C:{fs['C']}")

            # ✅ 早停判断逻辑
            current_kappa = system_metrics['Kappa']
            best_score = optimization_state["best_score"]
            patience_counter = optimization_state["patience_counter"]

            # 检查是否有提升
            if current_kappa > best_score + Config.STOPPING_CRITERIA['min_delta']:
                # 🎉 新提升！
                optimization_state["best_score"] = current_kappa
                optimization_state["best_round"] = round_i
                optimization_state["best_prompts"] = current_prompts.copy()
                optimization_state["patience_counter"] = 0

                # 保存最佳提示词
                save_best_prompts(current_prompts, round_i, current_kappa)
                print(f"\n🏆 NEW BEST! Round {round_i} (Kappa: {current_kappa:.4f})")

            else:
                # ⏳ 无提升
                optimization_state["patience_counter"] += 1
                print(f"\n⏳ No improvement (Current: {current_kappa:.4f}, Best: {best_score:.4f})")
                print(f"   Patience: {optimization_state['patience_counter']}/{Config.STOPPING_CRITERIA['patience']}")

            # 检查终止条件
            if current_kappa >= Config.STOPPING_CRITERIA['target_kappa']:
                print(f"\n{'='*70}")
                print(f"🎉🎉🎉 TARGET REACHED! 🎉🎉🎉")
                print(f"System Kappa {current_kappa:.4f} >= {Config.STOPPING_CRITERIA['target_kappa']}")
                print(f"Optimization completed at Round {round_i}")
                print(f"{'='*70}")
                optimization_state["stopped"] = True
                break

            if optimization_state["patience_counter"] >= Config.STOPPING_CRITERIA['patience']:
                print(f"\n{'='*70}")
                print(f"⏹️ EARLY STOPPING TRIGGERED")
                print(f"No improvement for {optimization_state['patience_counter']} consecutive rounds")
                print(f"Best performance: Round {optimization_state['best_round']} (Kappa: {optimization_state['best_score']:.4f})")
                print(f"Loading best prompts for final use...")
                print(f"{'='*70}")

                # 回滚到最佳版本
                if optimization_state["best_prompts"]:
                    current_prompts = optimization_state["best_prompts"]
                optimization_state["stopped"] = True
                break

            # Phase 4: 词典更新
            for res in trace_results:
                lexicon_manager.check_and_buffer_new_words(res.get("Final_Output", {}))
            new_word_count = lexicon_manager.commit_new_words(round_i)
            if new_word_count > 0:
                print(f"\n📚 Lexicon Growth: +{new_word_count} words (Total: {sum(lexicon_manager.get_stats().values())})")

            # Phase 5: Meta-LLM优化（系统目标导向）
            for role in ["Annotator", "Reviewer", "Arbitrator"]:
                memory.record(role, round_i, role_metrics[role], current_prompts[role])
                # 扩大错题采样 + 按维度分组（解决Meta-LLM只看到3个随机错题的问题）
                bad_cases = evaluator.get_role_specific_bad_cases(trace_results, df_test, role, max_cases=15)
                # 提示词变更diff（解决Meta-LLM不知道自己上次改了什么的问题）
                prompt_diff = memory.compute_prompt_diff(role, round_i)
                # 历史失败策略警告
                failed_approaches = memory.get_failed_approaches(role)

                # 构建系统级优化指导
                system_guidance = f"""
=== SYSTEM FINAL TARGET ===
Current System Kappa: {system_metrics['Kappa']:.4f} (Target: {Config.STOPPING_CRITERIA['target_kappa']})
Current System F1:    {system_metrics['F1']:.4f} (Target: {Config.STOPPING_CRITERIA['target_f1']})
Current System Recall:    {system_metrics['Recall']:.4f}
Current System Precision: {system_metrics['Precision']:.4f}

OPTIMIZATION PRIORITY (Select based on above):
1. IF System Kappa < 0.80: Focus on Bilingual Format (RVS improvement)
2. IF System Recall < 0.92: Focus on Annotator (reduce False Negatives)
3. IF System Precision < 0.92: Focus on Reviewer (reduce False Positives)
4. IF Revision Rate > 20%: Focus on Annotator-Reviewer alignment

Current Revision Rate (dims rejected): {interaction_stats['revision_trigger_rate']:.1%}
Deadlock Rate (C-arbitrated dims): {dim_transitions['b2_rejected_after_a2']}/{dim_transitions['total_dimensions']} ({dim_transitions['c_used_dims']}/{dim_transitions['total_dimensions']})
"""

                role_report = f"""
=== ROLE PERFORMANCE: {role} ===
RSI: {role_metrics[role]['Composite_Score']:.4f}
Recall: {role_metrics[role]['Recall']:.4f} | Precision: {role_metrics[role]['Precision']:.4f}
RVS: {role_metrics[role]['RVS']:.4f} | Kappa: {role_metrics[role]['Kappa']:.4f}

{PromptLibrary.DIAGNOSIS_MAP[role]}

Top Issues:
{bad_cases}
"""

                round_info = f"Round: {round_i} / {Config.STOPPING_CRITERIA['max_rounds']}"
                history_warning = memory.get_history_warning(role)
                change_details = memory.get_last_change_details(role)
                system_trend = memory.get_system_trend(n_rounds=3)

                # ========== 构建带完整历史的 Meta-LLM 优化指令 ==========

                # 生成优化历史摘要（解决"没有记忆"的问题）
                opt_history_lines = []
                for h_rec in memory.system_history[:-1]:  # 除当前轮外的所有历史
                    opt_history_lines.append(
                        f"  Round {h_rec['round']}: "
                        f"Kappa={h_rec['kappa']:.4f} "
                        f"(Δ={h_rec['delta']:+.4f}) "
                        f"F1={h_rec['f1']:.4f} "
                        f"R={h_rec['recall']:.4f} "
                        f"P={h_rec['precision']:.4f}"
                    )
                if opt_history_lines:
                    opt_history_str = "Optimization History (previous rounds):\n" + "\n".join(opt_history_lines)
                else:
                    opt_history_str = "Optimization History: This is the first round. No history yet."

                # 提示词演变历史：Champion Prompt + 上一轮Prompt
                champion_info = ""
                if optimization_state["best_prompts"] and optimization_state["best_round"] > 0:
                    champion_prompt = optimization_state["best_prompts"].get(role, "")
                    if champion_prompt:
                        if round_i == optimization_state["best_round"]:
                            champion_info = (
                                f"\n=== 🏆 CHAMPION PROMPT (Round {optimization_state['best_round']}, "
                                f"Kappa={optimization_state['best_score']:.4f}) ===\n"
                                f"This IS the current best. Build on it.\n"
                                f"{champion_prompt}\n"
                            )
                        else:
                            # Compute diff from Champion to current
                            import difflib as df2
                            curr = current_prompts.get(role, "")
                            diff_lines = df2.unified_diff(
                                champion_prompt.splitlines(keepends=True),
                                curr.splitlines(keepends=True),
                                fromfile=f'Champion_R{optimization_state["best_round"]}',
                                tofile=f'Current_R{round_i}',
                                lineterm='')
                            champ_diff = ''.join(diff_lines)
                            if len(champ_diff) > 3000:
                                champ_diff = champ_diff[:3000] + "\n...(diff truncated, see full Champion below)"
                            champion_info = (
                                f"\n=== 🏆 CHAMPION PROMPT (Round {optimization_state['best_round']}, "
                                f"Kappa={optimization_state['best_score']:.4f}) ===\n"
                                f"Current Kappa: {system_metrics['Kappa']:.4f} "
                                f"(Δ from Champion: {system_metrics['Kappa'] - optimization_state['best_score']:+.4f})\n"
                                f"=== DIFF: Champion → Current ===\n{champ_diff}\n"
                                f"\n=== 🏆 CHAMPION PROMPT (FULL) ===\n{champion_prompt}\n"
                            )

                prompt_evolution = ""
                if round_i > 1:
                    prev_round = round_i - 1
                    prev_prompt_path = os.path.join(Config.HISTORY_ROOT, f"{role}_v{prev_round}.txt")
                    if os.path.exists(prev_prompt_path):
                        with open(prev_prompt_path, 'r', encoding='utf-8') as pf:
                            prev_content = pf.read()
                        prompt_evolution = (
                            f"\n=== YOUR PREVIOUS OUTPUT (Round {prev_round}) ===\n"
                            f"Performance change after that round: "
                            f"{memory.system_history[-2]['delta']:+.4f} Kappa\n"
                            f"Full text (first 1500 chars):\n{prev_content[:1500]}\n"
                        )

                # 尝试过但失败的策略（避免重复犯同样错误）
                failed_strategies = ""
                if round_i > 2:
                    # 检查是否有连续下降
                    declines = []
                    for i in range(1, len(memory.system_history)):
                        if memory.system_history[i]['delta'] < -0.005:
                            declines.append(f"  Round {memory.system_history[i]['round']}: "
                                           f"dropped {memory.system_history[i]['delta']:.4f}")
                    if declines:
                        failed_strategies = (
                            "\n⚠️ PREVIOUS FAILURES (strategies to avoid):\n"
                            + "\n".join(declines)
                            + "\nDo NOT repeat the same type of modification that caused these drops."
                        )

                # 维度过渡统计数据（帮助定位瓶颈）
                transition_debug = (
                    f"\n=== PIPELINE BOTTLENECK ANALYSIS ===\n"
                    f"B1 rejection rate: {dim_transitions['b1_rejected_dims']}/{dim_transitions['total_dimensions']} "
                    f"({dim_transitions['b1_pass_rate']:.1%})\n"
                    f"A2 actually changed: {dim_transitions['a2_actually_changed']} dimensions\n"
                    f"B2 re-rejected: {dim_transitions['b2_rejected_after_a2']}\n"
                    f"C used: {dim_transitions['c_used_dims']} times\n"
                    f"Final source: A1={dim_transitions['final_source_distribution'].get('A1',0)} "
                    f"A2={dim_transitions['final_source_distribution'].get('A2',0)} "
                    f"C={dim_transitions['final_source_distribution'].get('C',0)}\n"
                    f"Pipeline F1: A1={pipeline_contrib['f1_A1_only']:.4f} → "
                    f"+A2={pipeline_contrib['f1_A1_B1_A2']:.4f} → "
                    f"full={pipeline_contrib['f1_ABABC_full']:.4f}"
                )

                meta_user = f"""
=== CONTEXT ===
Target Role: {role}
{round_info}

{opt_history_str}
{failed_strategies}

{champion_info}

=== WHAT CHANGED LAST ROUND ===
{prompt_diff}
{failed_approaches}

{system_guidance}

{transition_debug}

{system_trend}

{role_report}

{change_details}

{history_warning}

{prompt_evolution}

=== CURRENT PROMPT (TO BE OPTIMIZED) ===
{current_prompts[role]}

=== OPTIMIZATION INSTRUCTION ===
Based on SYSTEM FINAL METRICS AND OPTIMIZATION HISTORY (not just current metrics), select strategy:
- Review what was done in previous rounds — do not repeat failed approaches
- If System needs Recall↑: Optimize Annotator for sensitivity
- If System needs Precision↑: Optimize Reviewer for strictness
- If System needs Kappa↑: Enforce Bilingual Format in ALL roles
- If B1 rejection rate is high: Improve Annotator first-pass accuracy
- If A2 rarely changes labels: Annotator is ignoring revision feedback
- If C is overused: Annotator and Reviewer definitions need alignment

Return optimized prompt in <H_RAMOS_PROMPT_V1> tags.
Document expected impact on System Kappa.
"""

                print(f"\n  🧠 Optimizing {role} (System-Targeted)...")

                try:
                    raw_response = await manager.call(
                        Config.META_MODEL,
                        [
                            {"role": "system", "content": PromptLibrary.META_SYSTEM_PROMPT},
                            {"role": "user", "content": meta_user}
                        ]
                    )

                    if isinstance(raw_response, dict) and "error" in raw_response:
                        print(f"    ❌ Meta-LLM API error: {raw_response['error']}")
                        continue

                    if isinstance(raw_response, dict):
                        raw_str = json.dumps(raw_response)
                    else:
                        raw_str = raw_response

                    cleaned_prompt = clean_and_save_prompt(raw_str, role, round_i)

                    if cleaned_prompt:
                        current_prompts[role] = cleaned_prompt
                        print(f"    ✅ {role} prompt updated")
                    else:
                        print(f"    ⚠️ Failed to process {role} prompt, keeping previous")

                except Exception as e:
                    print(f"    ❌ Optimization failed for {role}: {str(e)[:100]}")
                    import traceback
                    if Config.DEBUG_MODE:
                        traceback.print_exc()

            # 保存每轮完整报告
            report_path = os.path.join(Config.LOG_ROOT, f"Round_{round_i}_SystemReport.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "round": round_i,
                    "system_metrics": system_metrics,
                    "role_metrics": role_metrics,
                    "interaction_stats": interaction_stats,
                    "dimension_transitions": dim_transitions,
                    "pipeline_contribution": pipeline_contrib,
                    "lexicon_stats": lexicon_manager.get_stats(),
                    "optimization_state": {
                        "best_score": optimization_state["best_score"],
                        "best_round": optimization_state["best_round"],
                        "patience_counter": optimization_state["patience_counter"]
                    }
                }, f, ensure_ascii=False, indent=2, default=str)

    finally:
        # 最终报告
        final_stats = lexicon_manager.get_stats()
        print(f"\n{'=' * 70}")
        print(f"OPTIMIZATION COMPLETED")
        print(f"{'=' * 70}")
        print(f"Best Round: {optimization_state['best_round']}")
        print(f"Best System Kappa: {optimization_state['best_score']:.4f}")
        print(f"Target Kappa: {Config.STOPPING_CRITERIA['target_kappa']}")

        if optimization_state["best_score"] >= Config.STOPPING_CRITERIA['target_kappa']:
            print(f"Status: ✅ TARGET ACHIEVED")
        else:
            print(f"Status: ⏹️ EARLY STOP (Best effort)")

        print(f"\nBest Prompts Location: {Config.BEST_PROMPT_DIR}")
        print(f"Files:")
        for role in ["Annotator", "Reviewer", "Arbitrator"]:
            filename = f"{role}_BEST_R{optimization_state['best_round']}.txt"
            if os.path.exists(os.path.join(Config.BEST_PROMPT_DIR, filename)):
                print(f"  - {filename}")

        print(f"\nFinal Lexicon: {sum(final_stats.values())} words (+{sum(final_stats.values()) - sum(init_stats.values())})")
        print(f"{'=' * 70}")

        # ================================================================
        # AUTO-DEPLOY: 将最优提示词复制到 best_prompts_final/ 供05使用
        # 04输出: best_prompts/Annotator_BEST_R3.txt
        # 05期望: best_prompts_final/Annotator_best.txt
        # ================================================================
        deploy_dir = os.path.join(Config.BASE_ROOT, 'best_prompts_final')
        os.makedirs(deploy_dir, exist_ok=True)
        best_r = optimization_state['best_round']
        for role in ["Annotator", "Reviewer", "Arbitrator"]:
            src = os.path.join(Config.BEST_PROMPT_DIR, f"{role}_BEST_R{best_r}.txt")
            dst = os.path.join(deploy_dir, f"{role}_best.txt")
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"   ✅ Deployed {role}_best.txt (from Round {best_r})")
            else:
                print(f"   ⚠️ Source not found: {src}")
        print(f"   → Deployed to: {deploy_dir}")
        print(f"{'=' * 70}")

        # 保存最终总结
        summary_path = os.path.join(Config.BEST_PROMPT_DIR, "optimization_summary.json")

        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            return str(obj)

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "target_kappa": Config.STOPPING_CRITERIA['target_kappa'],
                "best_kappa": float(optimization_state["best_score"]),  # ✅ 强制转 float
                "best_round": int(optimization_state["best_round"]),  # ✅ 强制转 int
                "total_rounds": round_i,
                # ✅ 强制转 bool
                "target_achieved": bool(optimization_state["best_score"] >= Config.STOPPING_CRITERIA['target_kappa']),
                "initial_lexicon": init_stats,
                "final_lexicon": final_stats,
                "system_history": memory.system_history
            }, f, ensure_ascii=False, indent=2, default=convert_numpy)

        await manager.close()
        print("\n🔒 Resources cleaned up. Ready for production use.")


# ===================== 9. 程序入口 =====================

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Program interrupted by user")
        print(f"💡 Best result so far: Round {optimization_state['best_round']} (Kappa: {optimization_state['best_score']:.4f})")
    except Exception as e:
        print(f"\n❌ Fatal error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
