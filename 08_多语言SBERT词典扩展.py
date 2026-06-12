"""
08_多语言SBERT词典扩展
=====================
利用多语言 SBERT 将 LLM 自由生成的关键词自动匹配到标准化词典。

核心机制:
    1. 从 05 标注结果中提取三国所有唯一关键词
    2. 将词典锚点关键词编码为 768 维 SBERT 向量
    3. 对每个新关键词，找到余弦相似度最高的 Top-3 词典锚点
    4. 多锚点投票决定 (维度, L1, L2) 的归属
    5. 跨语言验证 (Cross-Language Validation)：如果在另一语言中也有高相似度匹配，加 0.05 分
    6. 按阈值分档：
       - effective_score ≥ 0.70 → 自动接受，写入词典
       - 0.50 ≤ score < 0.70 → 人工审核队列（HIGH/MEDIUM/LOW 三档）
       - score < 0.50 → 丢弃

注意（循环论证问题）:
    08 读取 05 的输出 → 扩充词典 → 07 使用扩充后的词典。
    但 07 中优先使用 StandardConcepts 列（已在 05 阶段标准化），
    词典扩充主要影响回退路径（raw keyword lookup）。
    匹配率的提升部分来自"词典自举"——先往词典里加词，再说匹配率提高了。

输入:  05 输出 final_annotation_results_complete/{CHI,JPN,KOR}/*_Detailed_Annotation.xlsx
       merged_dictionary_v2.json (基础词典)

输出:  merged_dictionary_v2.json (覆盖更新，旧版备份为 .backup_round_N)
       多语言匹配结果报告.csv (自动接受的关键词及匹配信息)
       多语言未匹配关键词报告.csv (需人工审核的关键词，含优先级)
"""

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import warnings
import os

warnings.filterwarnings('ignore')

# ========================== 1. 全局配置 ==========================

# 多语言 SBERT 模型路径（本地部署）
SBERT_PATH = r"D:\summer_research\models\paraphrase-multilingual-MiniLM-L12-v2"

# 输入: 基础词典
BASE_DICT_PATH = r"D:\summer_research\投稿\code_media\co-occurrence network\词典存储\merged_dictionary_v2.json"

# 输入: 三国标注结果（从 05 输出读取）
ANNOTATION_FILES = {
    "CHI": r"D:\summer_research\投稿\code_media\final_annotation_results_complete\CHI\CHI_Detailed_Annotation.xlsx",
    "JPN": r"D:\summer_research\投稿\code_media\final_annotation_results_complete\JPN\JPN_Detailed_Annotation.xlsx",
    "KOR": r"D:\summer_research\投稿\code_media\final_annotation_results_complete\KOR\KOR_Detailed_Annotation.xlsx"
}

# 输出: 扩充后的词典（覆盖更新，旧版自动备份）
EXPANDED_DICT_SAVE_PATH = r"D:\summer_research\投稿\code_media\co-occurrence network\词典存储\merged_dictionary_v2.json"

# 输出: 自动接受的关键词报告
MATCH_REPORT_SAVE_PATH = r"D:\summer_research\投稿\code_media\co-occurrence network\词典存储\多语言匹配结果报告.csv"

# 输出: 未匹配（需人工审核）的关键词报告
UNMATCHED_REPORT_SAVE_PATH = r"D:\summer_research\投稿\code_media\co-occurrence network\词典存储\多语言未匹配关键词报告.csv"

# 相似度阈值
SIMILARITY_THRESHOLD = 0.7  # 自动接受阈值（实际使用 AUTO_THRESHOLD=0.70）
AUTO_THRESHOLD = 0.70       # effective_score ≥ 此值 → 自动接受
REVIEW_THRESHOLD = 0.50     # effective_score ≥ 此值 → 进入人工审核队列
CROSS_LANG_BOOST = 0.05     # 跨语言验证加分
TOP_K = 3                   # 每词取 Top-3 锚点进行投票

# ========================== 2. 关键词提取 ==========================

def load_keywords_from_excel(file_path, lang_code):
    """从 05 标注结果 Excel 中提取所有维度的关键词。

    优先匹配包含 "Keywords" 的特定维度列（如 Perceived_Cause_Keywords），
    如果没有找到则回退到通用关键词列名。

    处理步骤:
        1. 找到所有含 "Keywords" 的列
        2. 逐列读取，统一分割符（逗号/分号），去空格
        3. 过滤无效值（"none", "over_annotation", "nan"）
        4. 去重后返回

    Args:
        file_path: Detailed_Annotation.xlsx 路径
        lang_code: 语言代码 (CHI/JPN/KOR)，仅用于日志

    Returns:
        唯一关键词列表
    """
    if not os.path.exists(file_path):
        print(f"⚠️ [跳过] 找不到文件: {file_path}")
        return []

    try:
        df = pd.read_excel(file_path)

        target_cols = []
        possible_names = ["关键词", "keywords", "keyword", "Keywords",
                          "Perceived_Cause_Keywords", "Symptom_Description_Keywords",
                          "Coping_and_Management_Keywords", "Perceived_Consequences_Keywords",
                          "Emotional_Expression_Keywords", "Social_Interaction_Keywords"]

        # 优先匹配包含 "Keywords" 的特定维度列
        target_cols.extend([c for c in df.columns if "Keywords" in c])

        # 如果没有找到特定列，回退到通用列名
        if not target_cols:
            target_cols = [c for c in df.columns if any(p in c for p in possible_names)]

        print(f"   [{lang_code}] 正在从以下列提取: {target_cols[:3]}...")

        all_kws = set()
        for col in target_cols:
            raw_data = df[col].dropna().astype(str).tolist()
            for item in raw_data:
                # 统一分割符（中文逗号、英文逗号、分号）
                item = item.replace("，", ",").replace(";", ",")
                parts = [p.strip() for p in item.split(",") if p.strip()]
                for p in parts:
                    if p.lower() not in ["none", "over_annotation", "nan"]:
                        all_kws.add(p)

        print(f"   [{lang_code}] 提取到 {len(all_kws)} 个唯一关键词")
        return list(all_kws)

    except Exception as e:
        print(f"❌ [{lang_code}] 读取失败: {e}")
        return []


# ========================== 3. 主程序 ==========================

def main():
    # --- A. 加载基础词典并构建反向索引 ---
    print("1. Loading base dictionary...")
    with open(BASE_DICT_PATH, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)

    # 构建: 关键词 → (维度, L1, L2) 索引
    dict_index = {}
    dict_corpus = []  # 所有锚点关键词列表（用于批量编码）

    # 格式兼容：支持 {"mapping_rules": [...]} 和直接列表两种格式
    if "mapping_rules" in dictionary:
        rules = dictionary["mapping_rules"]
    elif isinstance(dictionary, list):
        rules = dictionary
        dictionary = {"mapping_rules": dictionary}  # 归一化，确保写回时不报错
    else:
        raise ValueError(
            f"词典格式无法识别。期望 keys: ['mapping_rules'] 或直接列表，"
            f"实际 keys: {list(dictionary.keys())[:5]}"
        )

    for rule in rules:
        dim = rule.get("维度")
        l1 = rule.get("一级标签")
        l2 = rule.get("二级标签")
        for kw in rule.get("keywords", []):
            clean_kw = kw.strip()
            if clean_kw:
                dict_index[clean_kw] = (dim, l1, l2)
                dict_corpus.append(clean_kw)
    print(f"   Loaded {len(dict_corpus)} anchor keywords.")

    # --- B. 加载 SBERT 模型并编码词典锚点 ---
    print("\n2. Loading multilingual SBERT model...")
    model = SentenceTransformer(SBERT_PATH)
    print("   Encoding dictionary anchors...")
    dict_embeddings = model.encode(dict_corpus, convert_to_tensor=True, show_progress_bar=True)

    # --- C. 多语言逐词匹配与扩展 ---
    print("\n3. Processing multi-language data...")

    all_new_matches = []    # 自动接受的匹配结果
    all_review_matches = [] # 需人工审核的匹配结果

    # 第一遍: 从三国 Excel 中提取所有关键词（按语言分组）
    lang_keywords = {}
    for lang_code, file_path in ANNOTATION_FILES.items():
        kws = load_keywords_from_excel(file_path, lang_code)
        lang_keywords[lang_code] = kws
        print(f"   [{lang_code}] {len(kws)} keywords extracted.")

    # 第二遍: 对每个语言的关键词执行多锚点匹配 + 跨语言验证
    for lang_code, input_kws in lang_keywords.items():
        if not input_kws:
            continue
        print(f"\n--- Processing {lang_code} ({len(input_kws)} keywords) ---")

        # 批量编码当前语言的所有关键词
        input_embeddings = model.encode(input_kws, convert_to_tensor=True, show_progress_bar=False)
        # 计算余弦相似度矩阵: (N_keywords, N_anchors)
        cos_scores = util.cos_sim(input_embeddings, dict_embeddings)
        scores_np = cos_scores.cpu().numpy()

        for i, current_kw in enumerate(input_kws):
            # --- Step 1: Top-K 锚点检索 ---
            row_scores = scores_np[i]
            top_k_indices = np.argsort(row_scores)[-TOP_K:][::-1]  # 降序取 Top-K
            top_k_scores = row_scores[top_k_indices]

            best_idx = top_k_indices[0]
            best_score = float(top_k_scores[0])
            best_anchor = dict_corpus[best_idx]
            dim, l1, l2 = dict_index[best_anchor]

            # --- Step 2: 多锚点投票 ---
            # 在 Top-K 中统计每个 (维度, L1, L2) 出现的次数
            vote_counts = {}
            for ki in range(min(TOP_K, len(top_k_indices))):
                idx_k = top_k_indices[ki]
                score_k = float(top_k_scores[ki])
                if score_k < REVIEW_THRESHOLD:
                    break  # 相似度过低，不再参与投票
                anchor_k = dict_corpus[idx_k]
                dim_k, l1_k, l2_k = dict_index[anchor_k]
                key = (dim_k, l1_k, l2_k)
                vote_counts[key] = vote_counts.get(key, 0) + 1

            # 得票最多的 (维度, L1, L2) 作为最终归属
            voted_key = max(vote_counts, key=vote_counts.get) if vote_counts else (dim, l1, l2)
            vote_strength = vote_counts.get(voted_key, 1) if vote_counts else 1
            final_dim, final_l1, final_l2 = voted_key

            # 备选锚点字符串（用于报告）
            alt_str = "; ".join([
                f"{dict_corpus[idx]}({float(row_scores[idx]):.3f})"
                for idx in top_k_indices[1:min(3, len(top_k_indices))]
            ])

            # --- Step 3: 跨语言验证 ---
            # 在其他语言中搜索是否有相同 (dim, L1, L2) 的高相似度匹配 (≥0.60)
            cross_validated = False
            for other_code in lang_keywords:
                if other_code == lang_code:
                    continue
                other_kws = lang_keywords[other_code]
                if len(other_kws) == 0:
                    continue
                # 为性能考虑，只取每种语言前 30 个关键词做交叉验证
                sample_size = min(30, len(other_kws))
                other_emb = model.encode(
                    other_kws[:sample_size],
                    convert_to_tensor=True, show_progress_bar=False
                )
                other_scores = util.cos_sim(other_emb, dict_embeddings)
                other_np = other_scores.cpu().numpy()
                for oi in range(sample_size):  # 只遍历实际编码的词数
                    o_top = np.argmax(other_np[oi])
                    o_anchor = dict_corpus[o_top]
                    o_dim, o_l1, o_l2 = dict_index[o_anchor]
                    if (o_dim, o_l1, o_l2) == voted_key and float(other_np[oi][o_top]) >= 0.60:
                        cross_validated = True
                        break
                if cross_validated:
                    break

            # effective_score = best_score + 跨语言加成
            effective_score = best_score + (CROSS_LANG_BOOST if cross_validated else 0.0)

            # --- Step 4: 阈值判定 ---
            record = {
                "Language": lang_code,
                "Keyword": current_kw,
                "Best_Anchor": best_anchor,
                "Similarity": round(best_score, 4),
                "Effective_Score": round(effective_score, 4),
                "Vote_Strength": vote_strength,
                "Cross_Lang_Validated": cross_validated,
                "Suggested_Dimension": final_dim,
                "Suggested_Level1": final_l1,
                "Suggested_Level2": final_l2,
                "Alt_Anchors": alt_str
            }

            if effective_score >= AUTO_THRESHOLD:
                all_new_matches.append(record)
            elif effective_score >= REVIEW_THRESHOLD:
                # 根据分数细分为 HIGH / MEDIUM 优先级
                record["Review_Priority"] = "HIGH" if effective_score >= 0.60 else "MEDIUM"
                all_review_matches.append(record)
            else:
                record["Review_Priority"] = "LOW"
                all_review_matches.append(record)

    # --- D. 保存结果 ---
    print("\n4. Saving results...")

    # D1. 自动接受的关键词 → 写入词典
    df_matches = pd.DataFrame(all_new_matches)
    if not df_matches.empty:
        df_matches.to_csv(MATCH_REPORT_SAVE_PATH, index=False, encoding='utf-8-sig')
        print(f"   Auto-matched: {len(df_matches)} keywords")

        update_count = 0
        for _, row in df_matches.iterrows():
            td = row["Suggested_Dimension"]
            tl1 = row["Suggested_Level1"]
            tl2 = row["Suggested_Level2"]
            nw = row["Keyword"]
            # 将新关键词添加到对应的 mapping_rules 条目中
            for rule in dictionary["mapping_rules"]:
                if (rule.get("维度") == td and
                        rule.get("一级标签") == tl1 and
                        rule.get("二级标签") == tl2):
                    if nw not in rule["keywords"]:
                        rule["keywords"].append(nw)
                        update_count += 1
                    break

        print(f"   Dictionary: +{update_count} keywords")
        # 覆盖写入词典（旧版由 LexiconManager 的 commit_new_words 方法备份）
        with open(EXPANDED_DICT_SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=4)
    else:
        print("   No auto-matched keywords.")

    # D2. 需人工审核的关键词 → 按优先级排序后保存
    df_review = pd.DataFrame(all_review_matches)
    if not df_review.empty:
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        df_review["_sort"] = df_review["Review_Priority"].map(priority_order)
        df_review = df_review.sort_values(["_sort", "Effective_Score"], ascending=[True, False])
        df_review = df_review.drop(columns=["_sort"])
        df_review.to_csv(UNMATCHED_REPORT_SAVE_PATH, index=False, encoding='utf-8-sig')
        print(f"\n   Needs review: {len(df_review)} keywords")
        for pri in ["HIGH", "MEDIUM", "LOW"]:
            cnt = len(df_review[df_review["Review_Priority"] == pri])
            print(f"      {pri}: {cnt}")
    else:
        print("   No keywords need review.")


if __name__ == "__main__":
    main()
