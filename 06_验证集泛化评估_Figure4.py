"""
06_验证集泛化评估（Figure 4）
===========================
对应文章 Figure 4: 在独立验证集上评估优化后提示词的泛化性能。

此脚本在验证集（独立于训练/测试集的 60 条数据）上运行最终选定的提示词，
生成泛化性能报告和训练曲线可视化。

工作流程:
    1. 从 experiment_full_trace/ 读取训练日志，绘制 Kappa/F1 优化曲线
    2. 加载 best_prompts_final/ 下的最佳提示词
    3. 注入词典并在验证集上运行维度级 ABABC 流程
    4. 计算 Kappa/F1/Recall/Precision/Accuracy/Macro_F1
    5. 保存结果 Excel 和完整 Trace JSON

关键设计:
    - 并发限制降低至 5（验证集较小，以求稳定）
    - 使用 ababc_utils 共享模块（与 03/05 一致）
    - 金标准列名通过 Config.DIM_MAP 映射

输入:  验证数据/验证data.xlsx (60条独立验证集)
       best_prompts_final/*_best.txt (04 输出的最佳提示词)
       merged_dictionary_v2.json (词典)
       experiment_full_trace/Round_*_SystemReport.json (训练日志，用于画图)

输出:  final_verification_results/
       ├── Final_Metrics_Report.xlsx (验证集指标)
       ├── Final_Trace_Details.json (完整推理链路)
       └── training_curve.png (训练优化曲线)
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import re
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from aiohttp import ClientSession, ClientTimeout
from asyncio import Semaphore
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, accuracy_score
from ababc_utils import (
    parse_b_verdicts as _parse_b_verdicts,
    get_dim_label as _get_dim_label,
    get_dim_confidence as _get_dim_confidence,
    get_dim_reasoning as _get_dim_reasoning,
    detect_changed_dimensions as _detect_changed_dimensions,
    build_dim_trace_empty as _build_dim_trace_empty,
    build_final_from_dim_trace as _build_final_from_dim_trace,
)

# ===================== 1. 全局配置 =====================

class Config:
    # --- 路径配置 (请确认这些路径真实存在) ---
    BASE_ROOT = r'D:\summer_research\投稿\code_media'

    # 输入数据：请确认这里是你要跑的【验证集】或【测试集】
    INPUT_FILE = os.path.join(BASE_ROOT, r'all_data\验证数据\验证data.xlsx')

    # 词典文件
    LEXICON_FILE = os.path.join(BASE_ROOT, r'co-occurrence network\词典存储\merged_dictionary_v2.json')

    # 提示词目录：请确保你已经把 v3.txt 复制进去并改名为 _best.txt
    PROMPT_DIR = os.path.join(BASE_ROOT, 'best_prompts_final')

    # 训练日志目录 (用于画图)
    LOG_DIR = os.path.join(BASE_ROOT, 'experiment_full_trace')

    # 输出结果目录
    RESULT_DIR = os.path.join(BASE_ROOT, 'final_verification_results')
    FINAL_REPORT_FILE = os.path.join(RESULT_DIR, 'Final_Metrics_Report.xlsx')

    # --- API Keys (请填入你的真实 Key) ---
    KEYS = {
        "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
        "doubao": os.getenv("DOUBAO_API_KEY", ""),
        "qwen": os.getenv("QWEN_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
    }
    DOUBAO_EP_ID = os.getenv("DOUBAO_ENDPOINT_ID", "")

    # --- 模型路由 ---
    WORKER_MODELS = {
        "Annotator": "deepseek-chat",           # Role A
        "Reviewer": "doubao-1-5-pro-32k-250115",# Role B
        "Arbitrator": "qwen-max"                # Role C
    }

    # --- 维度映射 ---
    DIM_MAP = {
        "Perceived Cause": "cause",
        "Symptom Description": "symptom",
        "Perceived Consequences": "consequences",
        "Coping and Management": "coping",
        "Emotional Expression": "emotion",
        "Social Interaction": "social"
    }

    # --- 运行参数 ---
    CONCURRENCY_LIMIT = 5  # 降低并发以求稳
    TIMEOUT = 60           # 单次请求超时时间

# ===================== 2. 全局维度列表（从 DIM_MAP 派生）=====================
DIM_NAMES = list(Config.DIM_MAP.keys())

# ===================== 3. 异步 API 管理器 =====================

class AsyncLLMManager:
    def __init__(self):
        self.clients = {}
        self.sem = Semaphore(Config.CONCURRENCY_LIMIT)

        # 初始化 OpenAI 兼容客户端
        if Config.KEYS["deepseek"]:
            self.clients["deepseek-chat"] = AsyncOpenAI(
                api_key=Config.KEYS["deepseek"], base_url="https://api.deepseek.com/v1"
            )
        if Config.KEYS["qwen"]:
            self.clients["qwen-max"] = AsyncOpenAI(
                api_key=Config.KEYS["qwen"], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

        # 初始化 豆包 HTTP Session
        self.session = ClientSession(
            headers={"Authorization": f"Bearer {Config.KEYS['doubao']}"},
            timeout=ClientTimeout(total=Config.TIMEOUT)
        )

    async def call(self, model_key, messages):
        """统一调用接口"""
        async with self.sem:
            try:
                if "doubao" in model_key:
                    return await self._call_doubao(messages)

                client = self.clients.get(model_key)
                if not client:
                    return {"error": f"Client for {model_key} not initialized"}

                resp = await client.chat.completions.create(
                    model=model_key,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                return self._safe_json(resp.choices[0].message.content)
            except Exception as e:
                print(f"❌ API Error ({model_key}): {e}")
                return {"error": str(e)}

    async def _call_doubao(self, messages):
        """豆包专用调用"""
        payload = {
            "model": Config.DOUBAO_EP_ID,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        try:
            async with self.session.post("https://ark.cn-beijing.volces.com/api/v3/chat/completions", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._safe_json(data['choices'][0]['message']['content'])
                else:
                    return {"error": f"Doubao Error {resp.status}: {await resp.text()}"}
        except Exception as e:
            return {"error": str(e)}

    def _safe_json(self, text):
        """清洗并解析 JSON"""
        if not text: return {}
        try:
            # 去掉 Markdown 代码块符号
            clean_text = re.sub(r'```(?:json)?\s*|\s*```', '', text.strip())
            return json.loads(clean_text)
        except:
            return {"error": "JSON Parse Failed", "raw": text}

    async def close(self):
        if not self.session.closed:
            await self.session.close()

# ===================== 4. 推理流水线（维度级ABABC）=====================

async def run_pipeline(manager, row_id, text, prompts, lexicon):
    """在单条帖子上运行维度级 ABABC 标注流程。

    与 ababc_utils.run_ababc_pipeline 不同，此函数直接内联实现，
    因为验证集脚本需要更灵活的 lexicon 注入和对特定管理器实例的耦合。

    流程: A1(全6维) → B1(维度审核) → A2(修订被拒维度) → B2(二次审核) → C(仲裁)

    Args:
        manager: AsyncLLMManager 实例
        row_id: 帖子 ID
        text: 帖子原文
        prompts: {"Annotator": str, "Reviewer": str, "Arbitrator": str}
        lexicon: 词典 JSON 字符串（注入到 Annotator 的 user message）

    Returns:
        trace dict (结构同 ababc_utils.run_ababc_pipeline 的返回值)
    """
    trace = {
        "ID": row_id, "Steps": {},
        "DimTrace": _build_dim_trace_empty(DIM_NAMES),
        "Exit_Stage": "Unknown", "Final_Output": {}
    }

    sys_a = prompts["Annotator"]
    sys_b = prompts["Reviewer"]

    # ==== A1: 首次标注全6维 ====
    msg_a1 = [{"role": "system", "content": sys_a},
              {"role": "user", "content": f"Text: {text}\nLexicon: {lexicon}"}]
    res_a1 = await manager.call(Config.WORKER_MODELS["Annotator"], msg_a1)
    trace["Steps"]["A1"] = res_a1

    if "error" in res_a1:
        return {**trace, "Exit_Stage": "Error_A1", "Final_Output": res_a1}

    for d in DIM_NAMES:
        trace["DimTrace"][d]["A1"] = res_a1.get(d, {})
        trace["DimTrace"][d]["final_label"] = _get_dim_label(res_a1, d)
        trace["DimTrace"][d]["final_source"] = "A1"
        trace["DimTrace"][d]["final_confidence"] = _get_dim_confidence(res_a1, d)

    # ==== B1: 维度级审核 ====
    msg_b1 = [{"role": "system", "content": sys_b},
              {"role": "user", "content": json.dumps({"text": text, "annotation": res_a1}, ensure_ascii=False)}]
    res_b1 = await manager.call(Config.WORKER_MODELS["Reviewer"], msg_b1)
    trace["Steps"]["B1"] = res_b1

    b1_verdicts = _parse_b_verdicts(res_b1, DIM_NAMES)
    for d in DIM_NAMES:
        v = b1_verdicts.get(d, {"approved": True, "feedback": ""})
        trace["DimTrace"][d]["B1_approved"] = v["approved"]
        trace["DimTrace"][d]["B1_feedback"] = v["feedback"]

    b1_rejected = [d for d in DIM_NAMES if not trace["DimTrace"][d]["B1_approved"]]
    if not b1_rejected:
        trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], DIM_NAMES)
        trace["Exit_Stage"] = "Consensus_R1"
        return trace

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
        f"PASSED dimensions ({len(DIM_NAMES) - len(b1_rejected)}): keep labels unchanged.\n"
        f"FAILED dimensions ({len(b1_rejected)}): REVISE based on B1 feedback.\n\n"
        f"{json.dumps(rejected_detail, ensure_ascii=False, indent=2)}\n\n"
        f"{json.dumps(res_a1, ensure_ascii=False, indent=2)}\n\n"
        f"Output ALL 6 dimensions. Passed → same label. Failed → corrected."
    )
    msg_a2 = [{"role": "system", "content": sys_a},
              {"role": "user", "content": f"{a2_instruction}\nLexicon: {lexicon}"}]
    res_a2 = await manager.call(Config.WORKER_MODELS["Annotator"], msg_a2)
    trace["Steps"]["A2"] = res_a2

    if "error" in res_a2:
        trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], DIM_NAMES)
        trace["Exit_Stage"] = "A2_Error"
        return trace

    for d in DIM_NAMES:
        trace["DimTrace"][d]["A2"] = res_a2.get(d, {})

    a2_changed, a2_changes = _detect_changed_dimensions(res_a1, res_a2, DIM_NAMES)
    for d in b1_rejected:
        if d in a2_changed:
            trace["DimTrace"][d]["final_label"] = _get_dim_label(res_a2, d)
            trace["DimTrace"][d]["final_source"] = "A2"
            trace["DimTrace"][d]["final_confidence"] = _get_dim_confidence(res_a2, d)

    # ==== B2: 仅检查A2改变了的维度 ====
    if a2_changed:
        b2_focus = {d: {
            "A1_label": a2_changes.get(d, {}).get("from", _get_dim_label(res_a1, d)),
            "A2_new_label": a2_changes.get(d, {}).get("to", _get_dim_label(res_a2, d)),
            "A2_reasoning": _get_dim_reasoning(res_a2, d),
            "B1_feedback": trace["DimTrace"][d]["B1_feedback"]
        } for d in a2_changed}

        b2_instruction = (
            f"Text: \"{text}\"\n\n"
            f"Review ONLY {len(a2_changed)} revised dimension(s):\n"
            f"{json.dumps(b2_focus, ensure_ascii=False, indent=2)}\n\n"
            f"Determine if each NEW label is correct. Output 'dimension_feedback' per dimension."
        )
        msg_b2 = [{"role": "system", "content": sys_b},
                  {"role": "user", "content": b2_instruction}]
        res_b2 = await manager.call(Config.WORKER_MODELS["Reviewer"], msg_b2)
        trace["Steps"]["B2"] = res_b2

        b2_verdicts = _parse_b_verdicts(res_b2, DIM_NAMES)
        for d in DIM_NAMES:
            v = b2_verdicts.get(d, {"approved": True, "feedback": ""})
            trace["DimTrace"][d]["B2_approved"] = v["approved"]
            trace["DimTrace"][d]["B2_feedback"] = v["feedback"]

        b2_rejected = [d for d in DIM_NAMES if not trace["DimTrace"][d]["B2_approved"]]
        if not b2_rejected:
            trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], DIM_NAMES)
            trace["Exit_Stage"] = "Consensus_R2"
            return trace

    # ==== C: 仅仲裁仍有争议的维度 ====
    still_unresolved = [d for d in DIM_NAMES
                        if trace["DimTrace"][d]["final_source"] == "A1"
                        and d in b1_rejected
                        and d not in a2_changed]
    contested_dims = list(set(
        [d for d in DIM_NAMES if trace["DimTrace"][d]["final_source"] is None
         or (trace["DimTrace"][d].get("B2_approved") is not None
             and not trace["DimTrace"][d]["B2_approved"])]
        + still_unresolved
    ))

    if contested_dims:
        arbitration_input = json.dumps({
            "text": text,
            "contested_dimensions": contested_dims,
            "history": {k: trace["Steps"][k] for k in ["A1", "B1", "A2", "B2"] if k in trace["Steps"]}
        }, ensure_ascii=False)
        sys_c = prompts["Arbitrator"]
        msg_c = [{"role": "system", "content": sys_c},
                 {"role": "user", "content": arbitration_input}]
        res_c = await manager.call(Config.WORKER_MODELS["Arbitrator"], msg_c)
        trace["Steps"]["C"] = res_c

        if "error" not in res_c:
            for d in contested_dims:
                c_dim_data = res_c.get(d, {}) if isinstance(res_c, dict) else {}
                if c_dim_data:
                    trace["DimTrace"][d]["C"] = c_dim_data
                    trace["DimTrace"][d]["final_label"] = _get_dim_label(res_c, d)
                    trace["DimTrace"][d]["final_source"] = "C"
                    trace["DimTrace"][d]["final_confidence"] = _get_dim_confidence(res_c, d)

    trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], DIM_NAMES)
    trace["Exit_Stage"] = "Arbitrated"
    return trace

# ===================== 5. 主流程逻辑（指标计算与可视化）=====================

def plot_training_history():
    """从 04 的训练日志 (experiment_full_trace/Round_*_SystemReport.json) 中
    读取各轮的 Kappa/F1 指标，绘制优化曲线并保存为 PNG。

    输出: final_verification_results/training_curve.png (300 dpi)
    """
    print(f"\n📊 Step 1: Visualizing Training History...")

    if not os.path.exists(Config.LOG_DIR):
        print("   ❌ Log directory not found. Skipping plot.")
        return

    # 读取日志文件
    history = []
    files = sorted([f for f in os.listdir(Config.LOG_DIR) if f.endswith('SystemReport.json')])

    for fname in files:
        try:
            with open(os.path.join(Config.LOG_DIR, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
                metrics = data.get('system_metrics', {})
                metrics['round'] = data.get('round', 0)
                history.append(metrics)
        except Exception:
            continue

    if not history:
        print("   ⚠️ No valid training logs found.")
        return

    df = pd.DataFrame(history)
    df = df.sort_values('round')

    # 绘图
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    plt.plot(df['round'], df['Kappa'], marker='o', linewidth=2, label='Kappa')
    plt.plot(df['round'], df['F1'], marker='s', linestyle='--', linewidth=2, label='F1 Score')

    plt.title('H-RAMOS Training Optimization Curve', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    img_path = os.path.join(Config.RESULT_DIR, 'training_curve.png')
    plt.savefig(img_path, dpi=300)
    print(f"   ✅ Training plot saved to {img_path}")
    plt.close()

async def final_verification_task():
    print(f"\n🧪 Step 2: Running Final Verification on Dataset...")
    print(f"   📂 Input: {Config.INPUT_FILE}")
    print(f"   📂 Prompts: {Config.PROMPT_DIR}")

    # 1. 加载提示词
    prompts = {}
    missing = []
    for role in ["Annotator", "Reviewer", "Arbitrator"]:
        path = os.path.join(Config.PROMPT_DIR, f"{role}_best.txt")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                prompts[role] = f.read()
        else:
            missing.append(f"{role}_best.txt")

    if missing:
        print(f"❌ CRITICAL ERROR: Missing prompt files in {Config.PROMPT_DIR}:")
        print(f"   {missing}")
        print("   👉 Please copy your best prompts (e.g., Annotator_best.txt) to this folder and rename them.")
        return

    print("   ✅ Prompts loaded successfully.")

    # 2. 加载词典
    lexicon_str = ""
    if os.path.exists(Config.LEXICON_FILE):
        with open(Config.LEXICON_FILE, 'r', encoding='utf-8') as f:
            lexicon_data = json.load(f)
            lexicon_str = json.dumps(lexicon_data, ensure_ascii=False)
            print(f"   📚 Lexicon loaded ({len(str(lexicon_str))} chars).")

    # 注入词典到 Annotator
    if "{lexicon}" in prompts["Annotator"]:
        prompts["Annotator"] = prompts["Annotator"].replace("{lexicon}", lexicon_str)

    # 3. 加载数据
    if not os.path.exists(Config.INPUT_FILE):
        print(f"❌ Input file not found: {Config.INPUT_FILE}")
        return

    df = pd.read_excel(Config.INPUT_FILE)
    df['ID'] = df['ID'].astype(str)
    print(f"   📊 Loaded {len(df)} samples.")

    # 4. 执行推理
    manager = AsyncLLMManager()
    tasks = []
    for _, row in df.iterrows():
        tasks.append(run_pipeline(manager, row['ID'], row['text'], prompts, lexicon_str))

    print("\n🚀 Starting Inference Pipeline...")
    results = await tqdm.gather(*tasks, desc="Verifying")
    await manager.close()

    # 5. 计算指标
    y_true = []
    y_pred = []

    print("\nCalculating Metrics...")
    for i, res in enumerate(results):
        row_id = str(res['ID'])
        gold_row = df[df['ID'] == row_id].iloc[0]
        final_output = res.get("Final_Output", {})

        if not isinstance(final_output, dict):
            # 如果输出出错，默认全 0
            for _ in Config.DIM_MAP:
                y_true.append(0)
                y_pred.append(0)
            continue

        for dim_full, dim_pre in Config.DIM_MAP.items():
            # Gold Label
            g = int(gold_row.get(f"{dim_pre}_label", 0))

            # Pred Label
            p_obj = final_output.get(dim_full, {})
            p_str = str(p_obj.get("label", "0")).lower()
            p = 1 if p_str in ['1', 'true'] else 0

            y_true.append(g)
            y_pred.append(p)

    # 6. 生成报告
    accuracy_val = accuracy_score(y_true, y_pred)
    macro_f1_val = f1_score(y_true, y_pred, average='macro', zero_division=0)

    metrics = {
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_val,
        "Macro_F1": macro_f1_val,
        "Samples": len(y_true)
    }

    print("\n" + "="*50)
    print("FINAL VERIFICATION RESULTS (Validation Set)")
    print("="*50)
    print(f"Kappa:     {metrics['Kappa']:.4f}")
    print(f"F1-Score:  {metrics['F1']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Macro_F1:  {metrics['Macro_F1']:.4f}")
    print("="*50)

    # 保存详细结果到 Excel
    # 转换 numpy 类型防止报错
    def clean_type(v):
        if isinstance(v, (np.floating, float)): return float(v)
        if isinstance(v, (np.integer, int)): return int(v)
        return v

    report_df = pd.DataFrame([metrics])
    report_df.to_excel(Config.FINAL_REPORT_FILE, index=False)
    print(f"📄 Report saved to: {Config.FINAL_REPORT_FILE}")

    # 保存详细的 Trace 用于人工检查
    trace_file = os.path.join(Config.RESULT_DIR, "Final_Trace_Details.json")
    with open(trace_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"📄 Trace details saved to: {trace_file}")

# ===================== 6. 程序入口 =====================

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    # 1. 画之前的训练曲线
    plot_training_history()

    # 2. 跑验证集
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(final_verification_task())