"""
02_单模型三提示词基线测试（Table 1）
================================
对应文章 Table 1: 三种提示词策略 × 三种 LLM 的单模型标注性能对比。

实验设计:
    - 3 种提示词: Direct (直接标注) / RULE (规则引导) / COT (思维链)
    - 3 种模型: DeepSeek-V3 / Doubao-Pro-1.5 / Qwen-Max
    - 在 100 条人工标注测试集上运行，计算各维度 + 总体的 F1/Kappa/RVS

关键设计决策:
    - 使用 OpenAI SDK 兼容接口（非 H-RAMOS 多智能体流程）
    - system role 放提示词指令，user role 放帖子正文（而非全部塞进 user）
    - 健壮 JSON 解析：依次尝试直接解析 → 清洗 markdown → 正则提取
    - 缓存机制：按 (prompt_name, model_name) 缓存 API 结果，支持断点续传

输入:  投稿/code_media/all_data/test_data/测试数据.xlsx (100条人工标注)
输出:  投稿/code_media/all_data/test_data/三提示词对比结果.xlsx
        - Sheet "原始标注明细": 每帖每模型每维度的标签/置信度/推理
        - Sheet "详细指标":   Prompt×Model×Dimension 级别的 F1/Kappa/RVS
        - Sheet "F1对比":     Model×Prompt 的 F1 透视表
"""

import pandas as pd
import numpy as np
import json
import os
import re
from openai import OpenAI  # 同步版 OpenAI SDK
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import (f1_score, cohen_kappa_score, accuracy_score,
                             precision_score, recall_score)
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局配置
# ==========================================

CONFIG = {
    # 三种提示词配置（从 JSON 文件加载 system_prompt_template）
    "prompts": [
        {"name": "Direct", "path": r"D:\summer_research\投稿\code_media\prompts\1_single_baselines\p1_direct.json"},
        {"name": "RULE",   "path": r"D:\summer_research\投稿\code_media\prompts\1_single_baselines\p2_rule.json"},
        {"name": "COT",    "path": r"D:\summer_research\投稿\code_media\prompts\1_single_baselines\p3_cot.json"}
    ],

    "test_data_path": r"D:\summer_research\投稿\code_media\all_data\test_data\测试数据.xlsx",
    "output_excel_path": r"D:\summer_research\投稿\code_media\all_data\test_data\三提示词对比结果.xlsx",

    # 三种模型的 API 配置
    "models": {
        "DeepSeek": {
            "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "base_url": "https://api.deepseek.com/v1",
            "model_name": "deepseek-chat"
        },
        "Doubao": {
            "api_key": os.getenv("DOUBAO_API_KEY", ""),
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model_name": os.getenv("DOUBAO_ENDPOINT_ID", "")  # 豆包使用 endpoint ID
        },
        "Aliyun": {
            "api_key": os.getenv("QWEN_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model_name": "qwen-max"
        }
    },

    # 金标准 Excel 列名映射: CSM维度 → (标签列, 推理列)
    "column_mapping": {
        "Perceived Cause":          {"label": "cause_label",       "reason": "cause_reasoning"},
        "Symptom Description":      {"label": "symptom_label",     "reason": "symptom_reasoning"},
        "Perceived Consequences":   {"label": "consequences_label","reason": "consequences_reasoning"},
        "Coping and Management":    {"label": "coping_label",      "reason": "coping_reasoning"},
        "Emotional Expression":     {"label": "emotion_label",     "reason": "emotion_reasoning"},
        "Social Interaction":       {"label": "social_label",      "reason": "social_reasoning"}
    },

    "core_dimensions": [
        "Perceived Cause", "Symptom Description", "Perceived Consequences",
        "Coping and Management", "Emotional Expression", "Social Interaction"
    ],

    # SBERT 模型路径（本地部署）
    "sbert_model": r"D:\summer_research\models\paraphrase-multilingual-MiniLM-L12-v2",
    "use_cache": True,
    "cache_dir": r"D:\summer_research\投稿\code_media\annotation_cache"
}

if CONFIG["use_cache"]:
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)

# OpenAI 客户端池（按 base_url 缓存，避免重复创建）
CLIENTS = {}


def get_client(model_config):
    """按 base_url 获取或创建 OpenAI 客户端实例。
    同一 base_url 的请求复用同一个客户端，减少连接开销。
    """
    cache_key = model_config['base_url']
    if cache_key not in CLIENTS:
        CLIENTS[cache_key] = OpenAI(
            api_key=model_config['api_key'],
            base_url=model_config['base_url']
        )
    return CLIENTS[cache_key]


# ==========================================
# 2. 提示词加载
# ==========================================

def load_prompt_from_json(prompt_file_path):
    """从 JSON 文件中加载 system_prompt_template。

    期望 JSON 格式: {"system_prompt_template": "..."}
    """
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"提示词文件不存在：{prompt_file_path}")

    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompt_json = json.load(f)

    if "system_prompt_template" not in prompt_json:
        raise KeyError("提示词文件必须包含'system_prompt_template'字段")

    return prompt_json["system_prompt_template"]


# ==========================================
# 3. API 调用与 JSON 解析
# ==========================================

def call_model(model_config, user_post, system_prompt_template):
    """使用 OpenAI SDK 调用单个模型进行标注。

    关键设计:
        - system role: 提示词指令（保持干净，仅含任务描述）
        - user role: 帖子正文 + 输出格式要求
        - temperature=0.0: 确定性输出
        - response_format={"type": "json_object"}: 强制 JSON 输出

    Args:
        model_config: 模型的 API 配置字典
        user_post: 待标注的帖子文本
        system_prompt_template: 提示词模板（不含帖子内容）

    Returns:
        解析后的 dict，或 None（API Key 未配置时），或 {"error": ...}
    """
    # 检查 API Key 是否已配置（简单判断：包含中文"你的"即未配置）
    if "你的" in model_config['api_key']:
        print(f"      ⚠️ API Key未配置，跳过")
        return None

    system_content = system_prompt_template

    try:
        client = get_client(model_config)

        # 用户正文放在 user role，system role 仅含任务指令
        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Text to analyze:\n\n{user_post}\n\n请根据上述指令分析文本，返回JSON格式结果。"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=60
        )

        content = response.choices[0].message.content
        return safe_parse_json(content)

    except Exception as e:
        print(f"      ⚠️ API调用失败: {str(e)[:80]}")
        return None


def safe_parse_json(text):
    """健壮 JSON 解析：依次尝试多种策略，最大限度提取有效 JSON。

    策略（按优先级）:
        1. 直接 json.loads (最理想的情况)
        2. 清洗 markdown 代码块标记 (```json ... ```) 后解析
        3. 正则提取首对完整花括号内容后解析
        4. 全部失败 → 返回 {"error": "parse_failed"}
    """
    if not text or not isinstance(text, str):
        return {"error": "empty_response"}

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    try:
        clean = re.sub(r'```json\s*|\s*```', '', text.strip())
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r'\{[\s\S]*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass

    return {"error": "parse_failed", "raw": text[:200]}


# ==========================================
# 4. RVS (Reasoning Vector Similarity) 计算
# ==========================================

def calculate_rvs(pred_reason, gold_reason, sbert_model):
    """计算预测推理与金标准推理之间的语义向量相似度。

    使用 SBERT 将两端推理文本编码为 768 维向量，计算余弦相似度。
    RVS 仅在 pred=1 且 gold=1 时计算（即双方都认为该维度存在时）。

    Args:
        pred_reason: LLM 输出的推理文本
        gold_reason: 人工标注的推理文本
        sbert_model: 已加载的 SentenceTransformer 模型

    Returns:
        余弦相似度 (0.0-1.0)，计算失败时返回 0.0
    """
    if not pred_reason or not gold_reason or pd.isna(pred_reason) or pd.isna(gold_reason):
        return 0.0
    try:
        pred_emb = sbert_model.encode(str(pred_reason), convert_to_tensor=True)
        gold_emb = sbert_model.encode(str(gold_reason), convert_to_tensor=True)
        similarity = util.cos_sim(pred_emb, gold_emb).item()
        return round(float(similarity), 4)
    except:
        return 0.0


# ==========================================
# 5. 指标计算
# ==========================================

def compute_all_metrics(annotation_results, gold_data, sbert_model):
    """计算所有 (Prompt, Model, Dimension) 组合的分类指标和 RVS。

    计算流程:
        1. 按 (Prompt, Model) 分组
        2. 对每组内的每个维度，匹配金标准计算 F1/Precision/Recall/Kappa/Accuracy
        3. 对 pred=1 & gold=1 的维度计算 RVS
        4. 汇总 Overall 指标（所有维度合并）

    Args:
        annotation_results: DataFrame，含每帖每模型的预测结果
        gold_data: DataFrame，人工金标准
        sbert_model: SBERT 模型（用于 RVS）

    Returns:
        DataFrame: 每行一个 (Prompt, Model, Dimension) 的完整指标
    """
    dim_metrics_list = []
    groups = annotation_results.groupby(["Prompt", "Model"])

    for (prompt_name, model_name), group in groups:
        print(f"   计算指标：Prompt={prompt_name}, Model={model_name}, 样本数={len(group)}")

        overall_y_true, overall_y_pred = [], []

        for dim in CONFIG["core_dimensions"]:
            col_info = CONFIG["column_mapping"][dim]
            label_col, reason_col = col_info["label"], col_info["reason"]

            dim_y_true, dim_y_pred, dim_rvs = [], [], []

            for _, row in group.iterrows():
                sample_id = str(row["ID"])
                gold_row = gold_data[gold_data["ID"].astype(str) == sample_id]

                if gold_row.empty:
                    continue

                gold_label = int(gold_row[label_col].values[0])
                gold_reason = gold_row[reason_col].values[0] if reason_col in gold_row.columns else ""

                # 从预测结果中提取该模型对该维度的标注
                pred_key = f"{model_name}_{dim}"
                if pred_key not in row or not isinstance(row[pred_key], dict):
                    continue

                pred = row[pred_key]
                if "label" not in pred:
                    continue

                pred_label = int(pred["label"])
                pred_reason = pred.get("reasoning", "")

                dim_y_true.append(gold_label)
                dim_y_pred.append(pred_label)
                overall_y_true.append(gold_label)
                overall_y_pred.append(pred_label)

                # RVS: 仅在双方都标注为正例时计算
                if pred_label == 1 and gold_label == 1:
                    rvs = calculate_rvs(pred_reason, gold_reason, sbert_model)
                    if rvs > 0:
                        dim_rvs.append(rvs)

            if len(dim_y_true) > 0:
                dim_metrics = {
                    "Prompt": prompt_name, "Model": model_name, "Dimension": dim,
                    "Samples": len(dim_y_true), "Positives": sum(dim_y_true),
                    "Accuracy": round(accuracy_score(dim_y_true, dim_y_pred), 4),
                    "Precision": round(precision_score(dim_y_true, dim_y_pred, zero_division=0), 4),
                    "Recall": round(recall_score(dim_y_true, dim_y_pred, zero_division=0), 4),
                    "F1": round(f1_score(dim_y_true, dim_y_pred, zero_division=0), 4),
                    "Kappa": round(cohen_kappa_score(dim_y_true, dim_y_pred), 4) if len(set(dim_y_true)) > 1 else 0.0,
                    "Avg_RVS": round(np.mean(dim_rvs), 4) if dim_rvs else 0.0,
                    "RVS_Count": len(dim_rvs)
                }
                dim_metrics_list.append(dim_metrics)

        # 汇总 Overall（所有维度合并）
        if len(overall_y_true) > 0:
            overall_metrics = {
                "Prompt": prompt_name, "Model": model_name, "Dimension": "Overall",
                "Samples": len(overall_y_true), "Positives": sum(overall_y_true),
                "Accuracy": round(accuracy_score(overall_y_true, overall_y_pred), 4),
                "Precision": round(precision_score(overall_y_true, overall_y_pred, zero_division=0), 4),
                "Recall": round(recall_score(overall_y_true, overall_y_pred, zero_division=0), 4),
                "F1": round(f1_score(overall_y_true, overall_y_pred, zero_division=0), 4),
                "Kappa": round(cohen_kappa_score(overall_y_true, overall_y_pred), 4) if len(
                    set(overall_y_true)) > 1 else 0.0,
                "Avg_RVS": round(np.mean([m["Avg_RVS"] for m in dim_metrics_list if
                                          m["Prompt"] == prompt_name and m["Model"] == model_name]), 4),
                "RVS_Count": sum([m["RVS_Count"] for m in dim_metrics_list if
                                  m["Prompt"] == prompt_name and m["Model"] == model_name])
            }
            dim_metrics_list.append(overall_metrics)

    return pd.DataFrame(dim_metrics_list)


# ==========================================
# 6. 缓存管理（支持断点续传）
# ==========================================

def get_cache_path(prompt_name, model_name):
    """获取缓存文件路径。文件名: cache_{prompt_name}_{model_name}.json"""
    if not CONFIG["use_cache"]:
        return None
    return os.path.join(CONFIG["cache_dir"], f"cache_{prompt_name}_{model_name}.json")


def load_cached_results(prompt_name, model_name):
    """从磁盘加载已有的标注结果。"""
    cache_path = get_cache_path(prompt_name, model_name)
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_cached_results(results, prompt_name, model_name):
    """将标注结果保存到磁盘缓存。每 10 条批量写入。"""
    cache_path = get_cache_path(prompt_name, model_name)
    if cache_path:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


# ==========================================
# 7. 主流程
# ==========================================

def main():
    print("🚀 三提示词 × 三模型 对比实验（OpenAI SDK版）")
    print("=" * 80)

    # --- 加载测试数据 ---
    try:
        df_test = pd.read_excel(CONFIG["test_data_path"])
        df_test["ID"] = df_test["ID"].astype(str)
        print(f"✅ 加载测试数据：{len(df_test)}条样本")
    except Exception as e:
        print(f"❌ 测试数据加载失败：{e}")
        return

    # --- 加载 SBERT 模型（用于 RVS 计算）---
    print("✅ 加载SBERT模型...")
    try:
        sbert = SentenceTransformer(CONFIG["sbert_model"])
    except Exception as e:
        print(f"⚠️ SBERT加载失败：{e}")
        sbert = None

    all_annotation_results = []

    # --- 外层循环: 遍历 3 种提示词 ---
    for prompt_config in CONFIG["prompts"]:
        prompt_name = prompt_config["name"]
        prompt_path = prompt_config["path"]

        print(f"\n{'=' * 60}")
        print(f"📝 提示词：{prompt_name}")
        print(f"{'=' * 60}")

        try:
            system_prompt_template = load_prompt_from_json(prompt_path)
        except Exception as e:
            print(f"❌ 加载失败：{e}")
            continue

        # --- 内层循环: 遍历 3 种模型 ---
        for model_name, model_config in CONFIG["models"].items():
            print(f"\n🔍 模型：{model_name}")

            # 检查 API Key 是否配置
            if "你的" in model_config["api_key"]:
                print("   ⚠️ 跳过：API Key未配置")
                continue

            # 加载已有缓存（支持断点续传）
            cached_results = load_cached_results(prompt_name, model_name)
            cached_ids = {r["ID"] for r in cached_results}
            if cached_results:
                print(f"   💾 缓存：{len(cached_results)}条")
                all_annotation_results.extend(cached_results)

            current_batch = []  # 批量写入缓冲区

            # --- 遍历每条测试数据 ---
            for idx, row in df_test.iterrows():
                sample_id = row["ID"]
                if sample_id in cached_ids:
                    continue  # 命中缓存，跳过 API 调用

                if idx % 5 == 0:
                    print(f"   进度：{idx + 1}/{len(df_test)}")

                user_post = row["text"]

                # 调用 API 进行标注
                pred = call_model(model_config, user_post, system_prompt_template)

                if pred is None or "error" in pred:
                    error_msg = pred.get("error", "unknown") if pred else "None"
                    print(f"   ❌ ID={sample_id} 失败: {error_msg}")
                    continue

                # 整理结果：为每个维度提取 label/confidence/reasoning
                sample_result = {
                    "ID": sample_id,
                    "text": user_post,
                    "Model": model_name,
                    "Prompt": prompt_name
                }

                valid = True
                for dim in CONFIG["core_dimensions"]:
                    if dim not in pred:
                        valid = False
                        continue

                    dim_data = pred[dim]
                    if isinstance(dim_data, dict):
                        sample_result[f"{model_name}_{dim}"] = {
                            "label": int(dim_data.get("label", 0)),
                            "confidence": float(dim_data.get("confidence", 2.0)),
                            "reasoning": str(dim_data.get("reasoning", ""))
                        }
                    else:
                        # 兼容非 dict 格式（直接是 label 值）
                        sample_result[f"{model_name}_{dim}"] = {
                            "label": int(dim_data),
                            "confidence": 2.0,
                            "reasoning": ""
                        }

                if valid:
                    current_batch.append(sample_result)
                    all_annotation_results.append(sample_result)

                    # 每 10 条批量写入缓存
                    if len(current_batch) >= 10:
                        existing = load_cached_results(prompt_name, model_name)
                        existing.extend(current_batch)
                        save_cached_results(existing, prompt_name, model_name)
                        cached_ids.update([r["ID"] for r in current_batch])
                        current_batch = []

            # 写入剩余批次
            if current_batch:
                existing = load_cached_results(prompt_name, model_name)
                existing.extend(current_batch)
                save_cached_results(existing, prompt_name, model_name)

    if not all_annotation_results:
        print("❌ 无有效结果")
        return

    # --- 计算指标 ---
    df_anno = pd.DataFrame(all_annotation_results)
    print(f"\n✅ 标注完成：总计{len(df_anno)}条")

    print("✅ 计算指标...")
    df_metrics = compute_all_metrics(df_anno, df_test, sbert)

    # --- 保存 Excel（三个 Sheet）---
    try:
        with pd.ExcelWriter(CONFIG["output_excel_path"], engine="openpyxl") as writer:
            # Sheet 1: 原始标注明细
            df_anno.to_excel(writer, sheet_name="原始标注明细", index=False)
            # Sheet 2: 详细指标（Prompt×Model×Dimension）
            df_metrics.to_excel(writer, sheet_name="详细指标", index=False)

            # Sheet 3: F1 透视表（Model × Prompt）
            df_overall = df_metrics[df_metrics["Dimension"] == "Overall"]
            pivot_f1 = df_overall.pivot_table(index="Model", columns="Prompt", values="F1").round(4)
            pivot_f1.to_excel(writer, sheet_name="F1对比")

            best = df_overall.loc[df_overall["F1"].idxmax()]
            print(f"\n🎉 完成！最优组合：{best['Prompt']}+{best['Model']} (F1={best['F1']:.4f})")

        print(f"📊 结果保存在：{CONFIG['output_excel_path']}")

    except Exception as e:
        print(f"❌ 保存失败：{e}")


if __name__ == "__main__":
    main()
