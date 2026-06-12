"""
07_PMI共现网络构建（Figures 5-7）
===============================
对应文章 Figures 5-7: 三国牙痛叙事的 CSM 维度共现网络。

功能:
    1. 从 05 的标注结果中提取每个维度的关键词/标准概念
    2. 通过词典将自由关键词映射到标准化的 L2 概念
    3. 计算概念间的 PMI (Pointwise Mutual Information) 权重
    4. 过滤低置信度（conf<4.0）、低共现（<6次）、低PMI（≤1.0）的边
    5. 过滤全局频率 <3 的外围节点
    6. 输出 Gephi 兼容的 nodes.csv 和 edges.csv

对应文章 Methods 2.5 节:
    - PMI > 1.0 且 co-occurrence ≥ 6 的边被保留
    - 全局频率 < 3 的概念节点被排除

输入:  05 输出目录 final_annotation_results_complete/{CHI,JPN,KOR}/*_Detailed_Annotation.xlsx
        merged_dictionary_v2.json (词典，用于关键词→L2概念映射)

输出:  gephi_nodes_L2_{国家}.csv  (Id, Label, Dimension, Level1_Category, Frequency, Size)
       gephi_edges_L2_PMI_{国家}.csv  (Source, Target, Type, Weight, Co_Occurrence)
"""

import pandas as pd
import json
import re
import math
import itertools
from collections import Counter, defaultdict
import os
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 路径配置 =================

# 共享词典路径（三国共用同一词典）
DICT_PATH = r"D:\summer_research\投稿\code_media\co-occurrence network\词典存储\merged_dictionary_v2.json"

# 标注结果基础路径（05 输出）
BASE_PATH = r"D:\summer_research\投稿\code_media\final_annotation_results_complete"

# Gephi 网络文件输出目录
OUTPUT_BASE = r"D:\summer_research\投稿\code_media\co-occurrence network\Network_Files"

# 三国数据配置: 输入路径 → 输出 nodes/edges 路径
COUNTRY_CONFIG = {
    "CHI": {
        "data_path": os.path.join(BASE_PATH, "CHI", "CHI_Detailed_Annotation.xlsx"),
        "output_nodes": os.path.join(OUTPUT_BASE, "gephi_nodes_L2_CHI.csv"),
        "output_edges": os.path.join(OUTPUT_BASE, "gephi_edges_L2_PMI_CHI.csv")
    },
    "JPN": {
        "data_path": os.path.join(BASE_PATH, "JPN", "JPN_Detailed_Annotation.xlsx"),
        "output_nodes": os.path.join(OUTPUT_BASE, "gephi_nodes_L2_JPN.csv"),
        "output_edges": os.path.join(OUTPUT_BASE, "gephi_edges_L2_PMI_JPN.csv")
    },
    "KOR": {
        "data_path": os.path.join(BASE_PATH, "KOR", "KOR_Detailed_Annotation.xlsx"),
        "output_nodes": os.path.join(OUTPUT_BASE, "gephi_nodes_L2_KOR.csv"),
        "output_edges": os.path.join(OUTPUT_BASE, "gephi_edges_L2_PMI_KOR.csv")
    }
}

# ================= 2. 列映射配置 =================

# 原始关键词列 → 目标维度名（用于词典查询）
# 注意：维度名需与词典中的 "维度" 字段一致
COL_MAPPING = {
    "Perceived_Cause_Keywords":          ["Perceived_Cause（感知病因）"],
    "Symptom_Description_Keywords":      ["Symptom_Description（症状描述）"],
    "Coping_and_Management_Keywords":    ["Coping_and_Management（应对方式）"],
    "Perceived_Consequences_Keywords":   ["Perceived_Consequences（感知后果）"],
    "Emotional_Expression_Keywords":     ["Emotional_Expression（情绪表达）"],
    "Social_Interaction_Keywords":       ["Social_Interaction（互动需求）"]
}

# 标准化概念列（由 KeywordStandardizer 预处理，优先使用）
# 这些列直接包含 L2 概念名，无需词典查询
STD_CONCEPT_MAPPING = {
    "Perceived_Cause_StandardConcepts":          ["Perceived_Cause（感知病因）"],
    "Symptom_Description_StandardConcepts":      ["Symptom_Description（症状描述）"],
    "Coping_and_Management_StandardConcepts":    ["Coping_and_Management（应对方式）"],
    "Perceived_Consequences_StandardConcepts":   ["Perceived_Consequences（感知后果）"],
    "Emotional_Expression_StandardConcepts":     ["Emotional_Expression（情绪表达）"],
    "Social_Interaction_StandardConcepts":       ["Social_Interaction（互动需求）"]
}


# ================= 3. 词典加载 =================

def load_lookup_dict(dict_path):
    """加载共享词典，构建关键词→(维度, L1, L2) 的快速查询表。

    词典格式兼容两种:
        1. {"mapping_rules": [{"维度": ..., "一级标签": ..., "二级标签": ..., "keywords": [...]}, ...]}
        2. 直接列表 [{"维度": ..., ...}, ...]

    Args:
        dict_path: merged_dictionary_v2.json 的路径

    Returns:
        lookup: {(维度名, 关键词): (清洗后维度名, L1标签, L2标签)}
    """
    with open(dict_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lookup = {}
    # 正则：去除中英文括号及其内容（如 "Perceived_Cause（感知病因）" → "Perceived_Cause"）
    pattern = re.compile(r'\s*[（\(].*?[）\)]')

    # 格式探测
    if "mapping_rules" in data:
        rules = data["mapping_rules"]
    elif isinstance(data, list):
        rules = data
    else:
        raise ValueError(
            f"词典格式无法识别。期望 keys: ['mapping_rules'] 或直接列表，"
            f"实际 keys: {list(data.keys())[:5]}"
        )

    for rule in rules:
        dim = rule.get("维度")
        clean_dim = re.sub(pattern, '', dim).strip() if dim else "Unknown"
        clean_l1 = re.sub(pattern, '', rule.get("一级标签", "Unknown")).strip()
        clean_l2 = re.sub(pattern, '', rule.get("二级标签", "Unknown")).strip()

        for kw in rule.get("keywords", []):
            kw = kw.strip()
            lookup[(dim, kw)] = (clean_dim, clean_l1, clean_l2)

    print(f"✅ 词典加载完成，准备映射网络...")
    return lookup


# ================= 4. 单国网络构建 =================

def build_network_for_country(country_code, config, lookup):
    """为一个国家构建 CSM 概念共现网络。

    处理流程:
        1. 读取 Detailed_Annotation.xlsx
        2. 逐帖提取概念标签：
           a. 优先使用 StandardConcepts 列（已预标准化的 L2 概念）
           b. 回退到原始 Keywords 列（通过词典映射到 L2 概念）
        3. 统计概念共现 (co-occurrence) 和文档频率 (document frequency)
        4. 计算 PMI 权重并过滤
        5. 输出 Gephi 兼容的 nodes.csv 和 edges.csv

    过滤阈值（与文章 Methods 2.5 节一致）:
        - 标注置信度 < 4.0 → 跳过
        - 概念全局频率 < 3 → 排除节点
        - 边 PMI ≤ 1.0 → 排除
        - 边共现频次 < 6 → 排除

    Args:
        country_code: 国家代码 (CHI/JPN/KOR)
        config: 包含 data_path, output_nodes, output_edges 的字典
        lookup: 词典查询表

    Returns:
        bool: 成功返回 True，失败返回 False
    """
    data_path = config["data_path"]
    output_nodes = config["output_nodes"]
    output_edges = config["output_edges"]

    print(f"\n{'=' * 60}")
    print(f"🌏 开始处理国家: {country_code}")
    print(f"📂 数据路径: {data_path}")
    print(f"{'=' * 60}")

    if not os.path.exists(data_path):
        print(f"❌ 错误: 文件不存在 {data_path}，跳过该国家")
        return False

    try:
        df = pd.read_excel(data_path)
        print(f"📊 成功读取 {len(df)} 条帖子")
    except Exception as e:
        print(f"❌ 读取Excel失败: {e}")
        return False

    # 初始化统计容器
    post_tags_list = []          # 每帖的概念集合列表（用于共现计数）
    node_attributes = {}         # 节点属性: {concept: {Id, Label, Dimension, Level1_Category}}
    tag_doc_freq = Counter()     # 概念文档频率
    skipped_low_conf = 0         # 被置信度过滤跳过的计数
    skipped_no_dict = 0          # 词典未覆盖的关键词计数
    std_concept_hits = 0         # StandardConcepts 命中计数

    print(f"⏳ 正在扫描帖子构建共现关系...")

    # --- 逐帖提取概念 ---
    for idx, row in df.iterrows():
        current_post_tags = set()

        # 第一轮: 优先尝试 StandardConcepts 列（预标准化的 L2 概念）
        for std_col, target_dims in STD_CONCEPT_MAPPING.items():
            dim_prefix = std_col.replace("_StandardConcepts", "")
            conf_col = f"{dim_prefix}_Confidence"

            # 置信度过滤: confidence < 4.0 → 跳过该维度
            confidence = row.get(conf_col, 0)
            try:
                confidence = float(confidence)
            except:
                confidence = 0
            if confidence < 4.0:
                skipped_low_conf += 1
                continue

            cell_val = row.get(std_col)
            if pd.isna(cell_val) or str(cell_val).strip() == "" or str(cell_val).lower() in ['none']:
                continue

            # StandardConcepts 中直接是 L2 概念名，按逗号分割
            concepts = [c.strip() for c in str(cell_val).split(',') if c.strip()]
            for concept in concepts:
                # 从词典中查找该 L2 概念的维度和 L1 信息
                matched_dim = None
                matched_l1 = None
                for target_dim in target_dims:
                    for (d, kw), (cd, cl1, cl2) in lookup.items():
                        if cl2 == concept and d == target_dim:
                            matched_dim = cd
                            matched_l1 = cl1
                            break
                    if matched_dim:
                        break

                if matched_dim:
                    current_post_tags.add(concept)
                    if concept not in node_attributes:
                        node_attributes[concept] = {
                            "Id": concept, "Label": concept,
                            "Dimension": matched_dim, "Level1_Category": matched_l1
                        }
                    std_concept_hits += 1
                else:
                    # 概念不在词典中 → 仍然作为新节点加入
                    current_post_tags.add(concept)
                    if concept not in node_attributes:
                        node_attributes[concept] = {
                            "Id": concept, "Label": concept,
                            "Dimension": target_dims[0], "Level1_Category": "Standardized"
                        }
                    std_concept_hits += 1

        # 第二轮: 回退到原始 Keywords 列（仅对没有 StandardConcepts 的维度）
        for col_name, target_dims in COL_MAPPING.items():
            dim_prefix = col_name.replace("_Keywords", "")
            std_col = f"{dim_prefix}_StandardConcepts"
            conf_col = f"{dim_prefix}_Confidence"

            # 如果该维度的 StandardConcepts 已有值，跳过关键词查询
            std_val = row.get(std_col)
            if not pd.isna(std_val) and str(std_val).strip() not in ["", "none"]:
                continue

            # 置信度过滤
            confidence = row.get(conf_col, 0)
            try:
                confidence = float(confidence)
            except:
                confidence = 0
            if confidence < 4.0:
                skipped_low_conf += 1
                continue

            cell_val = row.get(col_name)
            if pd.isna(cell_val) or str(cell_val).strip() == "" or str(cell_val).lower() in ['none']:
                continue

            # 分割关键词，通过词典映射到 L2 概念
            keywords = [k.strip() for k in str(cell_val).replace('，', ',').split(',') if k.strip()]
            for kw in keywords:
                matched = False
                for target_dim in target_dims:
                    if (target_dim, kw) in lookup:
                        c_dim, c_l1, c_l2 = lookup[(target_dim, kw)]
                        current_post_tags.add(c_l2)
                        if c_l2 not in node_attributes:
                            node_attributes[c_l2] = {
                                "Id": c_l2, "Label": c_l2,
                                "Dimension": c_dim, "Level1_Category": c_l1
                            }
                        matched = True
                        break
                if not matched:
                    skipped_no_dict += 1  # 关键词未被词典覆盖，静默丢弃

        # 将当前帖的概念集合加入列表
        if len(current_post_tags) > 0:
            post_tags_list.append(list(current_post_tags))
            tag_doc_freq.update(current_post_tags)

    if not post_tags_list:
        print(f"⚠️ 警告: {country_code} 没有提取到任何有效标签")
        return False

    # --- 打印过滤统计 ---
    print(f"   📊 置信度过滤: 跳过 {skipped_low_conf} 个低置信度标注 (conf < 4.0)")
    if std_concept_hits > 0:
        print(f"   📊 标准化概念命中: {std_concept_hits} (优先使用StandardConcepts列)")
    if skipped_no_dict > 0:
        print(f"   ⚠️ 词典未覆盖: {skipped_no_dict} 个关键词被静默丢弃（建议检查词典覆盖率）")

    # ================= 计算 PMI 边权重 =================
    print(f"⏳ 正在计算 PMI 权重...")
    N = len(post_tags_list)  # 总帖子数
    pair_counts = Counter()

    # 统计概念对在同一帖中的共现次数
    for tags in post_tags_list:
        if len(tags) < 2:
            continue  # 单概念帖子不产生边
        for source, target in itertools.combinations(sorted(tags), 2):
            pair_counts[(source, target)] += 1

    # 计算 PMI 并过滤
    edges_data = []
    for (source, target), count_xy in pair_counts.items():
        count_x = tag_doc_freq[source]
        count_y = tag_doc_freq[target]

        # PMI = log2( P(x,y) / (P(x) * P(y)) )
        #     = log2( (N * count_xy) / (count_x * count_y) )
        # 过滤: 共现 ≥ 6 且 PMI > 1.0（与文章一致）
        if count_x > 0 and count_y > 0 and count_xy >= 6:
            pmi = math.log2((N * count_xy) / (count_x * count_y))
            if pmi > 1.0:
                edges_data.append({
                    "Source": source,
                    "Target": target,
                    "Type": "Undirected",
                    "Weight": round(pmi, 4),
                    "Co_Occurrence": count_xy
                })

    # ================= 保存 Nodes.csv =================
    nodes_rows = []
    nodes_excluded_low_freq = 0
    for tag, attr in node_attributes.items():
        freq = tag_doc_freq[tag]
        if freq < 3:
            nodes_excluded_low_freq += 1
            continue  # 排除全局频率 < 3 的外围概念
        attr['Frequency'] = freq
        # Gephi 节点大小: 基于频率的对数缩放
        attr['Size'] = 10 + math.log(freq + 1) * 5
        nodes_rows.append(attr)

    if nodes_excluded_low_freq > 0:
        print(f"   📊 低频过滤: 排除 {nodes_excluded_low_freq} 个低频节点 (freq < 3)")

    df_nodes = pd.DataFrame(nodes_rows)
    if not df_nodes.empty:
        df_nodes = df_nodes[['Id', 'Label', 'Dimension', 'Level1_Category', 'Frequency', 'Size']]
        try:
            df_nodes.to_csv(output_nodes, index=False, encoding='utf-8-sig')
            print(f"✅ [Nodes] 节点表已生成: {output_nodes} (共 {len(df_nodes)} 个节点)")
        except Exception as e:
            print(f"❌ 保存Nodes失败: {e}")
            return False
    else:
        print(f"⚠️ 警告: {country_code} 没有生成任何节点")
        return False

    # ================= 保存 Edges.csv =================
    df_edges = pd.DataFrame(edges_data)
    if not df_edges.empty:
        df_edges = df_edges.sort_values(by='Weight', ascending=False)
        try:
            df_edges.to_csv(output_edges, index=False, encoding='utf-8-sig')
            print(f"✅ [Edges] 边表已生成: {output_edges} (共 {len(df_edges)} 条边)")
        except Exception as e:
            print(f"❌ 保存Edges失败: {e}")
            return False
    else:
        print(f"⚠️ 警告: {country_code} 未生成任何边，可能是PMI过滤阈值过高或数据量太少")

    print(f"🎉 {country_code} 网络构建完成！")
    return True


# ================= 5. 主程序 =================

def analyze_all_countries():
    """批量处理三个国家，生成各自的 Gephi 网络文件。"""
    # 1. 加载共享词典（只加载一次）
    lookup = load_lookup_dict(DICT_PATH)

    # 2. 确保输出目录存在
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # 3. 逐国处理
    success_countries = []
    failed_countries = []

    for country_code, config in COUNTRY_CONFIG.items():
        success = build_network_for_country(country_code, config, lookup)
        if success:
            success_countries.append(country_code)
        else:
            failed_countries.append(country_code)

    # 4. 最终报告
    print(f"\n{'=' * 60}")
    print("📋 批量处理报告")
    print(f"{'=' * 60}")
    print(f"✅ 成功处理: {', '.join(success_countries) if success_countries else '无'}")
    if failed_countries:
        print(f"❌ 处理失败: {', '.join(failed_countries)}")
    print(f"\n💡 提示:")
    print(f"   - 各国网络文件已保存在: {OUTPUT_BASE}")
    print(f"   - 可在Gephi中分别打开 CHI/JPN/KOR 文件进行对比分析")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    analyze_all_countries()
