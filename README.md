# H-RAMOS: 多智能体LLM牙痛叙事跨文化分析

> Social Media Narratives of Dental Pain in China, Japan, and South Korea

## 环境配置

```bash
# 1. 复制环境变量模板
cp .env.example .env

# 2. 编辑 .env，填入真实 API Key
DEEPSEEK_API_KEY=sk-xxx          # https://platform.deepseek.com
DOUBAO_API_KEY=xxx               # https://console.volcengine.com/ark
DOUBAO_ENDPOINT_ID=ep-m-xxx      # 豆包推理端点ID
QWEN_API_KEY=sk-xxx              # https://dashscope.aliyun.com

# 3. 安装依赖
pip install pandas numpy openpyxl scikit-learn sentence-transformers openai aiohttp tqdm

# 4. 确保以下路径存在数据文件
#    - 投稿/code_media/all_data/test_data/测试数据.xlsx (100条人工标注)
#    - 投稿/code_media/all_data/验证数据/验证data.xlsx (验证集)
#    - 投稿/code_media/all_data/raw_data/*_combined_data_cleaned.xlsx (三国清洗数据)
#    - 投稿/code_media/best_prompts_final/ (Annotator/Reviewer/Arbitrator_best.txt)
#    - 投稿/code_media/prompts/ (基线提示词 + 角色提示词)
#    - models/paraphrase-multilingual-MiniLM-L12-v2/ (SBERT模型)
```

## 流水线

```
01_数据清洗 → 02_基线测试 + 03_排列组合 → 04_提示词优化 → 05_全量标注 → 06_验证集 + 08_词典扩展 → 07_共现网络
```

| 编号 | 文件 | 对应文章 | 输入 | 输出 |
|:----:|------|:-------:|------|------|
| 01 | `01_数据清洗与预处理.py` | 2.2 | 原始Excel | `*_cleaned.xlsx` |
| 02 | `02_单模型三提示词基线测试_Table1.py` | Table 1 | 测试数据100条 | `三提示词对比结果.xlsx` |
| 03 | `03_六种ABABC排列组合实验_Table2.py` | Table 2 | 测试数据100条 | `ABABC_6Dimensions_Results.xlsx` |
| 04 | `04_提示词自动优化流水线_Figure3.py` | Figure 3 | 测试数据100条 | `best_prompts_final/` + 优化日志 |
| 05 | `05_三国全量标注_维度级ABABC.py` | 全量标注 | 01输出 + 04输出 | `final_annotation_results_complete/` |
| 06 | `06_验证集泛化评估_Figure4.py` | Figure 4 | 验证数据 + 04输出 | `final_verification_results/` |
| 07 | `07_PMI共现网络构建_Figure5-7.py` | Figures 5-7 | 05输出 | Gephi CSV |
| 08 | `08_多语言SBERT词典扩展.py` | 词典 | 05输出 | `merged_dictionary_v2.json` |
| — | `ababc_utils.py` | 共享模块 | — | `parse_b_verdicts` + `run_ababc_pipeline` |

## 架构

- **标注模型**: DeepSeek-V3 (Annotator) → Doubao-Pro-1.5 (Reviewer) → Qwen-Max (Arbitrator)
- **优化模型**: DeepSeek-V3 (Meta-LLM)
- **嵌入模型**: paraphrase-multilingual-MiniLM-L12-v2 (SBERT)
- **CSM维度**: Perceived Cause, Symptom Description, Perceived Consequences, Coping and Management, Emotional Expression, Social Interaction

## ABABC 共享模块

`ababc_utils.py` 是 03/04/05/06 四个流水线文件的唯一 ABABC 实现入口。所有文件通过 `from ababc_utils import parse_b_verdicts` 统一解析 Reviewer 反馈，通过 `run_ababc_pipeline()` 统一执行 A1→B1→A2→B2→C 流程。

## 词典

360 条专家规则 × 6 维度，覆盖中/日/韩三语。`merged_dictionary_v2.json` 为基础词典 + SBERT 扩展后的合并版本。
