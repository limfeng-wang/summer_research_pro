"""
05_三国全量标注（维度级 ABABC）
=============================
对中/日/韩三国全部 12,082 条帖子执行完整的维度级 ABABC 标注流程。

流程概述:
    1. 加载 01 输出的清洗数据 (*_combined_data_cleaned.xlsx)
    2. 加载 04 输出的最佳提示词 (best_prompts_final/*_best.txt)
    3. 注入动态词典到 Annotator 提示词 ({lexicon} 占位符替换)
    4. 逐帖执行 A1→B1→A2→B2→C 维度级标注（含重试 + 限流）
    5. 缓存每帖完整 trace 到 annotation_cache_complete/{国家}_{ID}.json
    6. 输出三个层次的结果 Excel (详细 / 统计 / 摘要)

模型架构:
    - Annotator (A): deepseek-chat     — 高召回导向，首次标注 + 修订
    - Reviewer (B):  doubao-1-5-pro    — 高精确导向，维度级审核
    - Arbitrator (C): qwen-max         — 最终裁决，仅处理争议维度

关键特性:
    - 缓存续传: 已有缓存的帖子直接返回，支持中断后继续
    - 指数退避重试: 最多 4 次，限流自动等待
    - 批次冷却: 每 60 条暂停 3 秒避免 API 限流
    - 连接池管理: TCPConnector 复用，limit=100
    - 中国数据回退: CHI 主文件不存在时自动尝试 aligned_annotations_cleaned.xlsx

输入:  01 输出 *_cleaned.xlsx, 04 输出 best_prompts_final/*_best.txt
       merged_dictionary_v2.json (动态词典)

输出:  final_annotation_results_complete/
       ├── 00_Cross_Country_Summary.xlsx (三国对比汇总)
       ├── CHI/
       │   ├── CHI_Detailed_Annotation.xlsx    (每帖6维度完整标注+溯源)
       │   ├── CHI_Dimension_Statistics.xlsx   (维度级统计)
       │   └── CHI_Sample_Summary.xlsx         (样本级摘要)
       ├── JPN/ (同上)
       └── KOR/ (同上)
"""

import os
import json
import pandas as pd
import numpy as np
import asyncio
import re
import time
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from asyncio import Semaphore
from typing import Dict, List, Tuple
from ababc_utils import (
    parse_b_verdicts as _parse_b_verdicts,
    get_dim_label as _get_dim_label,
    get_dim_confidence as _get_dim_confidence,
    get_dim_reasoning as _get_dim_reasoning,
    detect_changed_dimensions as _detect_changed_dimensions,
    build_dim_trace_empty as _build_dim_trace_empty,
    build_final_from_dim_trace as _build_final_from_dim_trace,
    extract_trace_meta as _extract_trace_meta,
)


# ===================== 1. 完整配置 (Production Ready) =====================

class MultiCountryConfig:
    # ✅ 路径配置（根据您的要求）
    BASE_ROOT = r'D:\summer_research\投稿\code_media'
    PROMPT_DIR = os.path.join(BASE_ROOT, r'best_prompts_final')  # 您的提示词目录
    LEXICON_FILE = r'D:\summer_research\投稿\code_media\co-occurrence network\词典存储\merged_dictionary_v2.json'  # 词典路径（合并修订版）
    OUTPUT_ROOT = os.path.join(BASE_ROOT, r'final_annotation_results_complete')
    CACHE_DIR = os.path.join(BASE_ROOT, r'annotation_cache_complete')

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 国家配置
    COUNTRIES = {
        "CHI": {
            "name": "China",
            "file": os.path.join(BASE_ROOT, r'all_data\raw_data\CHI_combined_data_cleaned.xlsx'),
            "id_col": "ID",
            "text_col": "full_text",
            "language": "Chinese"
        },
        "JPN": {
            "name": "Japan",
            "file": os.path.join(BASE_ROOT, r'all_data\raw_data\JA_combined_data_cleaned.xlsx'),
            "id_col": "ID",
            "text_col": "full_text",
            "language": "Japanese"
        },
        "KOR": {
            "name": "Korea",
            "file": os.path.join(BASE_ROOT, r'all_data\raw_data\KO_combined_data_cleaned.xlsx'),
            "id_col": "ID",
            "text_col": "full_text",
            "language": "Korean"
        }
    }

    # ✅ 模型配置（ABABC流程）
    WORKER_MODELS = {
        "Annotator": "deepseek-chat",  # A: 标注
        "Reviewer": "doubao-1-5-pro-32k-250115",  # B: 审查（高精度）
        "Arbitrator": "qwen-max"  # C: 仲裁
    }

    # API Keys
    KEYS = {
        "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
        "qwen": os.getenv("QWEN_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")),
        "doubao": os.getenv("DOUBAO_API_KEY", "")
    }

    # 运行参数
    CONCURRENCY_LIMIT = 15  # 并发数
    TIMEOUT = 50  # 超时
    MAX_RETRIES = 4  # 最大重试
    RETRY_DELAY = 1  # 重试延迟基数
    BATCH_SIZE = 60  # 批大小
    BATCH_COOLDOWN = 3  # 批次冷却
    CONNECTION_LIMIT = 100  # 连接池

    # CSM维度列表（与您的提示词一致）
    DIMENSIONS = [
        "Perceived Cause",
        "Symptom Description",
        "Perceived Consequences",
        "Coping and Management",
        "Emotional Expression",
        "Social Interaction"
    ]


# ===================== 2. 异步模型管理器（健壮版）=====================

class AsyncLLMManager:
    """异步多模型管理器（生产级）。

    特性:
        - 滑动窗口速率限制（max_rpm=15）
        - 指数退避重试（最多 4 次，基数 2^attempt 秒）
        - 限流自动等待（429 响应时额外等 10*(attempt+1) 秒）
        - TCP 连接池复用（limit=100, per_host=33, DNS 缓存 300s）
        - 并发信号量控制（Semaphore, limit=15）
    """

    def __init__(self):
        self.keys = MultiCountryConfig.KEYS
        self.clients = {}
        self.sem = Semaphore(MultiCountryConfig.CONCURRENCY_LIMIT)
        self.request_times = []
        self.max_rpm = 15

        # DeepSeek Client
        self.clients["deepseek-chat"] = AsyncOpenAI(
            api_key=self.keys["deepseek"],
            base_url="https://api.deepseek.com/v1",
            timeout=MultiCountryConfig.TIMEOUT
        )

        # Qwen Client
        self.clients["qwen-max"] = AsyncOpenAI(
            api_key=self.keys["qwen"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=MultiCountryConfig.TIMEOUT
        )

        # Doubao Config（endpoint模式）
        self.doubao_config = {
            "api_key": self.keys["doubao"],
            "base_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            "model": os.getenv("DOUBAO_ENDPOINT_ID", "")
        }

        # 连接池配置
        connector = TCPConnector(
            limit=MultiCountryConfig.CONNECTION_LIMIT,
            limit_per_host=MultiCountryConfig.CONNECTION_LIMIT // 3,
            ttl_dns_cache=300,
            use_dns_cache=True,
            enable_cleanup_closed=True,
            force_close=False,
        )

        self.session = ClientSession(
            connector=connector,
            timeout=ClientTimeout(total=MultiCountryConfig.TIMEOUT),
            headers={
                "Authorization": f"Bearer {self.doubao_config['api_key']}",
                "Content-Type": "application/json"
            }
        )

    async def _check_rate_limit(self):
        """滑动窗口速率限制"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) >= self.max_rpm:
            wait_time = 60 - (now - self.request_times[0]) + 1
            if wait_time > 0:
                print(f"  ⏳ 速率保护: 等待{wait_time:.1f}秒...")
                await asyncio.sleep(wait_time)

    async def call_with_retry(self, model_key: str, system_prompt: str, user_content: str) -> str:
        """带重试和验证的调用"""
        for attempt in range(MultiCountryConfig.MAX_RETRIES):
            try:
                await self._check_rate_limit()
                result = await self.call(model_key, system_prompt, user_content)

                # 验证结果有效性
                if result and result != "{}" and "error" not in result.lower():
                    return result

                # 无效结果，指数退避
                wait = MultiCountryConfig.RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(wait)

            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "too many requests" in error_msg:
                    wait = 10 * (attempt + 1)
                    print(f"  ⚠️ 触发限流，等待{wait}秒...")
                    await asyncio.sleep(wait)
                elif attempt < MultiCountryConfig.MAX_RETRIES - 1:
                    wait = MultiCountryConfig.RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(wait)
                else:
                    print(f"  ❌ 最终失败: {str(e)[:100]}")
                    return "{}"

        return "{}"

    async def call(self, model_key: str, system_prompt: str, user_content: str) -> str:
        """统一调用入口"""
        async with self.sem:
            try:
                if model_key == "doubao-1-5-pro-32k-250115":
                    return await self._call_doubao(system_prompt, user_content)
                else:
                    return await self._call_openai(model_key, system_prompt, user_content)
            except Exception as e:
                raise e

    async def _call_openai(self, model: str, system: str, user: str) -> str:
        """OpenAI兼容接口调用"""
        client = self.clients[model]
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user + "\n\nOutput valid JSON only."}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        self.request_times.append(time.time())
        return resp.choices[0].message.content

    async def _call_doubao(self, system: str, user: str) -> str:
        """Doubao专用HTTP调用"""
        payload = {
            "model": self.doubao_config["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user + " Output valid JSON."}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        async with self.session.post(self.doubao_config["base_url"], json=payload) as resp:
            self.request_times.append(time.time())
            if resp.status == 200:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status == 429:
                raise Exception("Rate limit")
            else:
                error_text = await resp.text()
                raise Exception(f"HTTP {resp.status}: {error_text[:100]}")

    async def close(self):
        """清理资源"""
        for c in self.clients.values():
            await c.close()
        await self.session.close()


# ===================== 3. 工具函数（整合两段优势）=====================

def clean_json(text: str) -> dict:
    """
    ✅ 终极JSON解析器：处理Markdown、嵌套结构、XML标签
    """
    if not text or not isinstance(text, str):
        return {"error": "empty_response"}

    # 步骤1：去除Markdown代码块和XML标签
    cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', text.strip())
    cleaned = re.sub(r'<H_RAMOS_PROMPT_V1>|</H_RAMOS_PROMPT_V1>', '', cleaned, flags=re.IGNORECASE)

    # 步骤2：尝试直接解析
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        else:
            return {"error": "not_dict", "raw": str(parsed)[:200]}
    except json.JSONDecodeError:
        pass

    # 步骤3：使用栈匹配法提取最外层JSON（处理嵌套）
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
                            objects.append(s[start:i + 1])
                            start = -1
        return objects

    # 从长到短尝试解析（优先最完整的）
    json_candidates = extract_json_objects(cleaned)
    for candidate in sorted(json_candidates, key=len, reverse=True):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # 步骤4：如果都失败，返回错误信息
    return {
        "error": "parse_failed",
        "raw_snippet": cleaned[:200] + "..." if len(cleaned) > 200 else cleaned
    }


def clean_keywords_for_excel(keywords: list) -> str:
    """
    ✅ 整合第二段的优化：清洗关键词，保留CJK字符，去重排序
    """
    if not isinstance(keywords, list):
        return ""

    cleaned = set()
    for k in keywords:
        k = str(k).strip()
        # 保留CJK字符、英文、数字，去除特殊标点
        k = re.sub(r'[^\w\s\u4e00-\u9fa5\u3040-\u30ff\u3130-\u318f\uac00-\ud7af]', '', k)
        if len(k) >= 1:
            cleaned.add(k)

    return ", ".join(sorted(list(cleaned)))


def get_cache_path(country_code: str, sample_id: str) -> str:
    """✅ 优化：使用复合键避免冲突，但保持原始ID不变"""
    safe_id = re.sub(r'[^\w\-]', '_', str(sample_id))
    return os.path.join(MultiCountryConfig.CACHE_DIR, f"{country_code}_{safe_id}.json")


def load_country_data(country_code: str, config: dict) -> pd.DataFrame:
    """加载国家数据，保持原始ID。主路径不存在时自动回退到备用文件。"""
    filepath = config["file"]
    if not os.path.exists(filepath):
        print(f"⚠️  {country_code} 主文件不存在: {filepath}")
        # 回退逻辑：中国数据尝试 aligned_annotations_cleaned.xlsx
        if country_code == "CHI":
            fallback = os.path.join(os.path.dirname(filepath), "aligned_annotations_cleaned.xlsx")
            if os.path.exists(fallback):
                print(f"   🔄 回退至备用文件: {fallback}")
                filepath = fallback
            else:
                print(f"   ❌ 备用文件也不存在: {fallback}")
                return None
        else:
            return None

    print(f"\n📂 加载 {config['name']} ({country_code})...")
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
        return None

    id_col = config["id_col"]
    text_col = config["text_col"]

    if id_col not in df.columns:
        raise ValueError(f"{country_code}: 缺少 '{id_col}' 字段")
    if text_col not in df.columns:
        raise ValueError(f"{country_code}: 缺少 '{text_col}' 字段")

    # ✅ 关键：保持原始ID不变，仅转为字符串
    df = df.rename(columns={id_col: "ID", text_col: "full_text"})
    df['ID'] = df['ID'].astype(str)

    # 检查重复ID并警告（不自动修改）
    if df['ID'].duplicated().any():
        dup_count = df['ID'].duplicated().sum()
        print(f"   ⚠️  警告: 发现 {dup_count} 个重复 ID！缓存可能冲突")

    df['Country'] = country_code

    # 数据清洗
    null_count = df['full_text'].isna().sum()
    df = df.dropna(subset=['full_text'])
    df = df[df['full_text'].str.len() > 5]

    if null_count > 0:
        print(f"   ⚠️  过滤 {null_count} 条空/无效文本")

    print(f"   ✅ 有效样本: {len(df)} 条")
    return df


def load_optimized_prompts(prompt_dir: str) -> Dict[str, str]:
    """✅ 加载best_prompts_final目录下的提示词"""
    prompts = {}
    # 根据您的要求，加载best版本
    files = {
        "Annotator": "Annotator_best.txt",
        "Reviewer": "Reviewer_best.txt",
        "Arbitrator": "Arbitrator_best.txt"
    }

    print(f"\n📄 从 {prompt_dir} 加载提示词...")
    for role, filename in files.items():
        filepath = os.path.join(prompt_dir, filename)

        if not os.path.exists(filepath):
            print(f"   ❌ 未找到 {filename}")
            # 尝试查找任何匹配的txt文件
            import glob
            pattern = os.path.join(prompt_dir, f"{role}*.txt")
            matches = glob.glob(pattern)
            if matches:
                filepath = matches[0]
                print(f"   ✅ 使用替代文件: {os.path.basename(filepath)}")
            else:
                print(f"   ❌ 未找到任何{role}提示词，使用默认")
                prompts[role] = f"You are a {role}. Output valid JSON."
                continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # 简单清洗：去除XML标签（如果有）
            if '<H_RAMOS_PROMPT_V1>' in content.upper():
                content = re.sub(r'<H_RAMOS_PROMPT_V1>|</H_RAMOS_PROMPT_V1>', '', content, flags=re.IGNORECASE)

            prompts[role] = content
            print(f"   ✅ {role}: 已加载 ({len(content)} 字符)")

            # 验证Annotator包含词典占位符
            if role == "Annotator" and "{lexicon}" not in content:
                print(f"   🚨 警告: {role} 提示词缺少 {{lexicon}} 占位符！")

        except Exception as e:
            print(f"   ❌ {role} 读取失败: {e}")
            prompts[role] = f"You are a {role}. Output valid JSON."

    return prompts


# ===================== 4. 核心ABABC流程（维度级版本）=====================

async def annotate_sample(
        sample_id: str,
        text: str,
        country: str,
        manager: AsyncLLMManager,
        prompts: Dict[str, str],
        use_cache: bool = True
) -> Tuple[str, str, dict, str]:
    """对单条帖子执行完整的维度级 ABABC 标注流程。

    返回:
        (sample_id, country_code, final_output_dict, exit_stage_str)

    退出阶段:
        - Consensus_R1: B1 批准了全部 6 维度，A1 结果直接采纳
        - Consensus_R2: B2 批准了 A2 的所有修改
        - Arbitrated: C 仲裁了至少一个争议维度
        - Error_A1/A2: API 调用失败，降级处理
        - *_Cached: 从缓存直接加载（追加 _Cached 后缀）
    """
    dim_names = MultiCountryConfig.DIMENSIONS
    cache_file = get_cache_path(country, sample_id)

    # 1. 检查缓存（现在保存完整trace）
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if "Output" in cached and "Exit_Stage" in cached:
                return sample_id, country, cached["Output"], cached["Exit_Stage"] + "_Cached"
        except Exception as e:
            print(f"   ⚠️  缓存读取失败: {e}")

    trace = {
        "ID": sample_id,
        "Steps": {},
        "DimTrace": _build_dim_trace_empty(dim_names),
        "DimChanges": {},
        "Final_Output": {},
        "Exit_Stage": "Unknown"
    }

    annotator_prompt = prompts["Annotator"]

    # ==== A1: 首次标注全6维 ====
    res_a1_str = await manager.call_with_retry(
        MultiCountryConfig.WORKER_MODELS["Annotator"],
        annotator_prompt,
        f"Text: {text}"
    )
    res_a1 = clean_json(res_a1_str)
    trace["Steps"]["A1"] = res_a1

    if "error" in res_a1:
        trace["Exit_Stage"] = "Error_A1"
        _save_cache(cache_file, {"Output": res_a1, "Exit_Stage": "Error_A1", "Trace": trace})
        return sample_id, country, res_a1, "Error_A1"

    for d in dim_names:
        trace["DimTrace"][d]["A1"] = res_a1.get(d, {})
        trace["DimTrace"][d]["final_label"] = _get_dim_label(res_a1, d)
        trace["DimTrace"][d]["final_source"] = "A1"
        trace["DimTrace"][d]["final_confidence"] = _get_dim_confidence(res_a1, d)

    # ==== B1: 维度级审核 ====
    review_input = json.dumps({"text": text, "annotation": res_a1}, ensure_ascii=False)
    res_b1_str = await manager.call_with_retry(
        MultiCountryConfig.WORKER_MODELS["Reviewer"],
        prompts["Reviewer"],
        review_input
    )
    res_b1 = clean_json(res_b1_str)
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
        final_with_meta = dict(trace["Final_Output"])
        final_with_meta.update(_extract_trace_meta(trace["DimTrace"], dim_names))
        result = {"Output": final_with_meta, "Exit_Stage": "Consensus_R1", "Trace": trace}
        _save_cache(cache_file, result)
        return sample_id, country, final_with_meta, "Consensus_R1"

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
        f"Dimensions PASSED ({len(dim_names) - len(b1_rejected)}): keep their current labels unchanged.\n"
        f"Dimensions FAILED ({len(b1_rejected)}): REVISE based on B1 feedback.\n\n"
        f"=== B1 FEEDBACK FOR FAILED DIMENSIONS ===\n"
        f"{json.dumps(rejected_detail, ensure_ascii=False, indent=2)}\n\n"
        f"=== YOUR CURRENT ANNOTATION (A1) ===\n"
        f"{json.dumps(res_a1, ensure_ascii=False, indent=2)}\n\n"
        f"Output ALL 6 dimensions. Passed dimensions → same label as A1. "
        f"Failed dimensions → corrected label with improved reasoning."
    )
    res_a2_str = await manager.call_with_retry(
        MultiCountryConfig.WORKER_MODELS["Annotator"],
        annotator_prompt,
        a2_instruction
    )
    res_a2 = clean_json(res_a2_str)
    trace["Steps"]["A2"] = res_a2

    if "error" in res_a2:
        trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], dim_names)
        trace["Exit_Stage"] = "A2_Error"
        final_with_meta = dict(trace["Final_Output"])
        final_with_meta.update(_extract_trace_meta(trace["DimTrace"], dim_names))
        result = {"Output": final_with_meta, "Exit_Stage": "A2_Error", "Trace": trace}
        _save_cache(cache_file, result)
        return sample_id, country, final_with_meta, "A2_Error"

    for d in dim_names:
        trace["DimTrace"][d]["A2"] = res_a2.get(d, {})

    a2_changed, a2_changes = _detect_changed_dimensions(res_a1, res_a2, dim_names)
    trace["DimChanges"]["A1_to_A2"] = {k: v for k, v in a2_changes.items()}

    for d in b1_rejected:
        if d in a2_changed:
            trace["DimTrace"][d]["final_label"] = _get_dim_label(res_a2, d)
            trace["DimTrace"][d]["final_source"] = "A2"
            trace["DimTrace"][d]["final_confidence"] = _get_dim_confidence(res_a2, d)

    # ==== B2: 仅检查A2改变了的维度 ====
    if not a2_changed:
        # A2未改变任何维度（拒绝了但标注不变）→ 维持原判，去C
        pass
    else:
        b2_focus = {d: {
            "A1_label": a2_changes[d]["from"] if d in a2_changes else _get_dim_label(res_a1, d),
            "A2_new_label": a2_changes[d]["to"] if d in a2_changes else _get_dim_label(res_a2, d),
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
        res_b2_str = await manager.call_with_retry(
            MultiCountryConfig.WORKER_MODELS["Reviewer"],
            prompts["Reviewer"],
            b2_instruction
        )
        res_b2 = clean_json(res_b2_str)
        trace["Steps"]["B2"] = res_b2

        b2_verdicts = _parse_b_verdicts(res_b2, dim_names)
        for d in dim_names:
            v = b2_verdicts.get(d, {"approved": True, "feedback": ""})
            trace["DimTrace"][d]["B2_approved"] = v["approved"]
            trace["DimTrace"][d]["B2_feedback"] = v["feedback"]

        b2_rejected = [d for d in dim_names if not trace["DimTrace"][d]["B2_approved"]]
        if not b2_rejected:
            trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], dim_names)
            trace["Exit_Stage"] = "Consensus_R2"
            final_with_meta = dict(trace["Final_Output"])
            final_with_meta.update(_extract_trace_meta(trace["DimTrace"], dim_names))
            result = {"Output": final_with_meta, "Exit_Stage": "Consensus_R2", "Trace": trace}
            _save_cache(cache_file, result)
            return sample_id, country, final_with_meta, "Consensus_R2"

    # ==== C: 仅仲裁仍有争议的维度 ====
    contested_dims = [d for d in dim_names if trace["DimTrace"][d]["final_source"] is None
                      or (trace["DimTrace"][d].get("B2_approved") is not None
                          and not trace["DimTrace"][d]["B2_approved"])]
    # 也包含 A2 未实际改变但B1拒绝了的维度
    still_unresolved = [d for d in dim_names
                        if trace["DimTrace"][d]["final_source"] == "A1"
                        and d in b1_rejected
                        and d not in a2_changed]
    contested_dims = list(set(contested_dims + still_unresolved))

    if contested_dims:
        arbitration_input = json.dumps({
            "text": text,
            "contested_dimensions": contested_dims,
            "history": {
                "A1": {d: trace["DimTrace"][d]["A1"] for d in contested_dims if trace["DimTrace"][d]["A1"]},
                "B1_feedback": {d: trace["DimTrace"][d]["B1_feedback"] for d in contested_dims},
                "A2": {d: trace["DimTrace"][d]["A2"] for d in contested_dims if trace["DimTrace"][d]["A2"]},
                "B2_feedback": {d: trace["DimTrace"][d].get("B2_feedback", "") for d in contested_dims}
            }
        }, ensure_ascii=False)

        res_c_str = await manager.call_with_retry(
            MultiCountryConfig.WORKER_MODELS["Arbitrator"],
            prompts["Arbitrator"],
            arbitration_input
        )
        res_c = clean_json(res_c_str)
        trace["Steps"]["C"] = res_c

        if "error" not in res_c:
            for d in contested_dims:
                c_dim_data = res_c.get(d, {}) if isinstance(res_c, dict) else {}
                if c_dim_data:
                    trace["DimTrace"][d]["C"] = c_dim_data
                    trace["DimTrace"][d]["final_label"] = _get_dim_label(res_c, d)
                    trace["DimTrace"][d]["final_source"] = "C"
                    trace["DimTrace"][d]["final_confidence"] = _get_dim_confidence(res_c, d)
    else:
        res_c = {}

    # 构建最终输出
    trace["Final_Output"] = _build_final_from_dim_trace(trace["DimTrace"], dim_names)
    trace["Exit_Stage"] = "Arbitrated"

    final_with_meta = dict(trace["Final_Output"])
    final_with_meta.update(_extract_trace_meta(trace["DimTrace"], dim_names))
    result = {"Output": final_with_meta, "Exit_Stage": "Arbitrated", "Trace": trace}
    _save_cache(cache_file, result)
    return sample_id, country, final_with_meta, "Arbitrated"


def _save_cache(cache_file: str, result: dict):
    """保存缓存（含完整维度级trace）"""
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"   ⚠️  缓存保存失败: {e}")


# ===================== 5. 批量处理（精细控制）=====================

async def process_batches(
        df: pd.DataFrame,
        country_code: str,
        manager: AsyncLLMManager,
        prompts: Dict[str, str]
) -> List[dict]:
    """分批处理数据"""

    all_results = []
    total = len(df)
    batch_size = MultiCountryConfig.BATCH_SIZE

    for i in range(0, total, batch_size):
        batch_df = df.iloc[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"\n   📦 批次 {batch_num}/{total_batches} ({len(batch_df)}条)...")

        # 创建任务列表
        tasks = []
        for _, row in batch_df.iterrows():
            tasks.append(annotate_sample(
                row["ID"],
                str(row["full_text"]),
                country_code,
                manager,
                prompts
            ))

        # 执行并收集结果
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        for idx, result in enumerate(tqdm(batch_results, desc=f"保存", leave=False)):
            if isinstance(result, Exception):
                sample_id = batch_df.iloc[idx]["ID"]
                all_results.append({
                    "ID": sample_id,
                    "Country": country_code,
                    "Text": str(batch_df.iloc[idx]["full_text"]),
                    "Output": {"error": str(result)},
                    "Exit_Stage": "Error_Exception"
                })
            else:
                rid, country, out, stage = result
                all_results.append({
                    "ID": rid,
                    "Country": country,
                    "Text": str(batch_df.iloc[idx]["full_text"]),
                    "Output": out,
                    "Exit_Stage": stage
                })

        # 批次间冷却
        if i + batch_size < total:
            await asyncio.sleep(MultiCountryConfig.BATCH_COOLDOWN)

    return all_results


# ===================== 6. 结果保存（精细记录版）=====================

def save_country_results(results: List[dict], country_code: str, output_dir: str):
    """为一个国家生成三个层次的标注结果 Excel 文件。

    输出文件:
        1. {国家}_Detailed_Annotation.xlsx:
           每行一条帖子，包含:
           - 6 维度的 Label / Confidence / Reasoning / Keywords / Risk
           - 各维度的 Final_Source (A1/A2/C) 和 B1_Approved (True/False)
           - Overall_Certainty / Cultural_Bias_Risk (_meta 信息)

        2. {国家}_Dimension_Statistics.xlsx:
           每个维度的汇总统计:
           - Positive_Count / Positive_Rate_Percent
           - Avg_Confidence_All / Avg_Confidence_Positive
           - High_Confidence_Rate_Percent (conf ≥ 4.0)
           - Risk_Tagged_Count

        3. {国家}_Sample_Summary.xlsx:
           每条帖子的摘要:
           - Exit_Stage / Dimensions_Detected / Avg_Confidence
           - Risk_Tags_Count / Text_Length / Text_Preview

    Args:
        results: process_batches 返回的结果列表
        country_code: 国家代码 (CHI/JPN/KOR)
        output_dir: 输出根目录
    """

    country_dir = os.path.join(output_dir, country_code)
    os.makedirs(country_dir, exist_ok=True)

    # 1. 详细结果 (Detailed)
    detailed = []
    for res in results:
        row = {
            "ID": res["ID"],
            "Country": res["Country"],
            "Full_Text": res["Text"],
            "Exit_Stage": res["Exit_Stage"]
        }

        output = res["Output"]
        if not isinstance(output, dict):
            output = {}

        # 提取6个维度（与您提示词一致）
        for dim in MultiCountryConfig.DIMENSIONS:
            dim_data = output.get(dim, {}) if isinstance(output, dict) else {}
            prefix = dim.replace(" ", "_")

            # Label (支持多种true表示)
            label_val = dim_data.get("label", 0) if isinstance(dim_data, dict) else 0
            row[f"{prefix}_Label"] = 1 if str(label_val).lower() in ["1", "true", "yes"] else 0

            # Confidence (2.0-5.0)
            conf = dim_data.get("confidence", 0) if isinstance(dim_data, dict) else 0
            try:
                row[f"{prefix}_Confidence"] = float(conf)
            except:
                row[f"{prefix}_Confidence"] = 0

            # Reasoning (保留双语言格式)
            reasoning = dim_data.get("reasoning", "") if isinstance(dim_data, dict) else ""
            row[f"{prefix}_Reasoning"] = str(reasoning)

            # Keywords (使用清洗函数)
            keywords = dim_data.get("keywords", []) if isinstance(dim_data, dict) else []
            row[f"{prefix}_Keywords"] = clean_keywords_for_excel(keywords)

            # Risk标签
            risk = dim_data.get("_risk", "none") if isinstance(dim_data, dict) else "none"
            row[f"{prefix}_Risk"] = risk

            # 维度级溯源：该维度的最终来源（A1/A2/C）
            row[f"{prefix}_Final_Source"] = output.get("_final_source", {}).get(dim, "")

            # B1审核是否通过
            row[f"{prefix}_B1_Approved"] = output.get("_b1_approved", {}).get(dim, "")

        # _meta 信息
        if isinstance(output, dict) and "_meta" in output:
            meta = output["_meta"]
            row["Overall_Certainty"] = meta.get("overall_certainty", "")
            row["Cultural_Bias_Risk"] = meta.get("cultural_bias_risk", "")
        else:
            row["Overall_Certainty"] = ""
            row["Cultural_Bias_Risk"] = ""

        detailed.append(row)

    df_detailed = pd.DataFrame(detailed)
    path1 = os.path.join(country_dir, f"{country_code}_Detailed_Annotation.xlsx")
    df_detailed.to_excel(path1, index=False, engine='openpyxl')

    # 2. 维度统计 (Statistics)
    stats = []
    for dim in MultiCountryConfig.DIMENSIONS:
        prefix = dim.replace(" ", "_")
        labels = df_detailed[f"{prefix}_Label"]
        confs = df_detailed[f"{prefix}_Confidence"]

        pos_mask = labels == 1
        stats.append({
            "Dimension": dim,
            "Total_Samples": len(df_detailed),
            "Positive_Count": int(pos_mask.sum()),
            "Positive_Rate_Percent": round(pos_mask.mean() * 100, 1) if len(df_detailed) > 0 else 0,
            "Avg_Confidence_All": round(confs.mean(), 2) if len(confs) > 0 else 0,
            "Avg_Confidence_Positive": round(confs[pos_mask].mean(), 2) if pos_mask.sum() > 0 else 0,
            "High_Confidence_Rate_Percent": round((confs >= 4.0).mean() * 100, 1) if len(confs) > 0 else 0,
            "Risk_Tagged_Count": (df_detailed[f"{prefix}_Risk"] != "none").sum()
        })

    df_stats = pd.DataFrame(stats)
    path2 = os.path.join(country_dir, f"{country_code}_Dimension_Statistics.xlsx")
    df_stats.to_excel(path2, index=False, engine='openpyxl')

    # 3. 样本摘要 (Summary)
    summary = []
    for res in results:
        output = res["Output"]
        dim_count = 0
        avg_conf = 0
        risk_count = 0

        if isinstance(output, dict):
            dim_count = sum(1 for dim in MultiCountryConfig.DIMENSIONS
                            if isinstance(output.get(dim), dict) and
                            str(output.get(dim, {}).get("label")).lower() in ["1", "true"])

            confs = [output.get(dim, {}).get("confidence", 0) for dim in MultiCountryConfig.DIMENSIONS
                     if isinstance(output.get(dim), dict)]
            avg_conf = np.mean(confs) if confs else 0

            # 统计风险标签
            for dim in MultiCountryConfig.DIMENSIONS:
                dim_data = output.get(dim, {})
                if isinstance(dim_data, dict) and dim_data.get("_risk", "none") != "none":
                    risk_count += 1

        summary.append({
            "ID": res["ID"],
            "Country": res["Country"],
            "Exit_Stage": res["Exit_Stage"],
            "Dimensions_Detected": dim_count,
            "Avg_Confidence": round(avg_conf, 2),
            "Risk_Tags_Count": risk_count,
            "Text_Length": len(res["Text"]),
            "Text_Preview": res["Text"][:150] + "..." if len(res["Text"]) > 150 else res["Text"]
        })

    df_summary = pd.DataFrame(summary)
    path3 = os.path.join(country_dir, f"{country_code}_Sample_Summary.xlsx")
    df_summary.to_excel(path3, index=False, engine='openpyxl')

    print(f"   ✅ 已保存: 详细({len(df_detailed)}条) | 统计 | 摘要")
    return df_detailed, df_stats, df_summary


def generate_cross_country_report(all_results: Dict[str, List[dict]], output_dir: str):
    """生成跨国对比报告"""
    print("\n📊 生成跨国对比汇总...")

    summary_data = []

    for country_code, results in all_results.items():
        if not results:
            continue

        total = len(results)
        exit_stages = pd.Series([r["Exit_Stage"] for r in results]).value_counts().to_dict()

        valid_results = [r for r in results if not r["Exit_Stage"].startswith("Error")]

        # 计算平均维度数
        avg_dims = 0
        if valid_results:
            dims_per_sample = []
            for r in valid_results:
                output = r["Output"]
                if isinstance(output, dict):
                    count = sum(1 for dim in MultiCountryConfig.DIMENSIONS
                                if isinstance(output.get(dim), dict) and
                                str(output.get(dim, {}).get("label")).lower() in ["1", "true"])
                    dims_per_sample.append(count)
            avg_dims = np.mean(dims_per_sample) if dims_per_sample else 0

        summary_data.append({
            "Country_Code": country_code,
            "Country_Name": MultiCountryConfig.COUNTRIES[country_code]["name"],
            "Total_Samples": total,
            "Valid_Samples": len(valid_results),
            "Success_Rate_Percent": round(len(valid_results) / total * 100, 1) if total > 0 else 0,
            "Language": MultiCountryConfig.COUNTRIES[country_code]["language"],
            "Avg_Dimensions_Detected": round(avg_dims, 2),
            "Consensus_R1": exit_stages.get("Consensus_R1", 0),
            "Consensus_R2": exit_stages.get("Consensus_R2", 0),
            "Arbitrated": exit_stages.get("Arbitrated", 0) + exit_stages.get("Arbitrated_Fallback", 0),
            "Cached": exit_stages.get("Consensus_R1_Cached", 0) + exit_stages.get("Consensus_R2_Cached", 0),
            "Errors": sum(v for k, v in exit_stages.items() if k.startswith("Error")),
            "Error_Rate_Percent": round(sum(v for k, v in exit_stages.items() if k.startswith("Error")) / total * 100,
                                        1) if total > 0 else 0
        })

    df_cross = pd.DataFrame(summary_data)
    cross_path = os.path.join(output_dir, "00_Cross_Country_Summary.xlsx")
    df_cross.to_excel(cross_path, index=False, engine='openpyxl')

    print("\n" + "=" * 80)
    print("🌍 跨国对比汇总:")
    print(df_cross.to_string(index=False))
    print("=" * 80)

    return df_cross


# ===================== 7. 主程序（完整版）=====================

async def main():
    print("🚀 H-RAMOS Multi-Country Final Annotation - Complete Version")
    print(f"⚡ 配置: {MultiCountryConfig.CONCURRENCY_LIMIT}并发 | 批次{MultiCountryConfig.BATCH_SIZE}")
    print("📌 模型: DeepSeek(A) → Doubao(B) → Qwen(C)")
    print("🔧 特性: 精细记录 | 词典注入 | 双语言验证")
    print("=" * 80)

    manager = AsyncLLMManager()

    # 1. 加载词典
    lexicon_content = "{}"
    lexicon_data = {}
    if os.path.exists(MultiCountryConfig.LEXICON_FILE):
        try:
            with open(MultiCountryConfig.LEXICON_FILE, 'r', encoding='utf-8') as f:
                lexicon_data = json.load(f)
                lexicon_content = json.dumps(lexicon_data, ensure_ascii=False, indent=2)
            print(
                f"📚 加载词典: {len(lexicon_data)}个维度, {sum(len(v) for v in lexicon_data.values() if isinstance(v, list))}个术语")
        except Exception as e:
            print(f"⚠️  词典加载失败: {e}，使用空词典")
    else:
        print(f"⚠️  词典文件不存在: {MultiCountryConfig.LEXICON_FILE}，使用空词典")

    # 2. 加载提示词
    prompts = load_optimized_prompts(MultiCountryConfig.PROMPT_DIR)

    if not prompts:
        print("❌ 未能加载必要提示词，程序终止")
        return

    # 3. 注入词典到Annotator
    if "Annotator" in prompts:
        if "{lexicon}" in prompts["Annotator"]:
            prompts["Annotator"] = prompts["Annotator"].replace("{lexicon}", lexicon_content)
            print("   ✅ 词典已成功注入Annotator提示词")

            # 验证注入
            if lexicon_content[:50] in prompts["Annotator"]:
                print("   ✅ 注入验证通过")
        else:
            print("   🚨 警告: Annotator提示词缺少{lexicon}占位符，无法注入词典")
            print("      将使用静态提示词，可能影响标注质量")
    else:
        print("❌ 缺少Annotator提示词，程序终止")
        return

    # 验证Reviewer和Arbitrator
    for role in ["Reviewer", "Arbitrator"]:
        if role not in prompts:
            print(f"🚨 警告: 缺少{role}提示词，使用默认")
            prompts[role] = f"You are a {role}. Output valid JSON."

    # 4. 处理所有国家
    all_results = {}
    start_time = time.time()

    try:
        for country_code, country_config in MultiCountryConfig.COUNTRIES.items():
            print(f"\n{'=' * 80}")
            print(f"🚀 开始处理 {country_config['name']} ({country_code})")
            print(f"   文件: {os.path.basename(country_config['file'])}")
            print(f"{'=' * 80}")

            # 加载数据
            df = load_country_data(country_code, country_config)
            if df is None or len(df) == 0:
                print(f"   ⚠️  跳过 {country_code}")
                continue

            # 批处理
            results = await process_batches(df, country_code, manager, prompts)
            all_results[country_code] = results

            # 保存结果（生成3个文件）
            save_country_results(results, country_code, MultiCountryConfig.OUTPUT_ROOT)

            # 显示退出阶段统计
            exit_counts = pd.Series([r["Exit_Stage"] for r in results]).value_counts()
            print(f"\n📈 {country_code} 完成统计:")
            for stage, count in exit_counts.items():
                print(f"   {stage}: {count} ({count / len(results) * 100:.1f}%)")

        # 生成跨国报告
        if all_results:
            generate_cross_country_report(all_results, MultiCountryConfig.OUTPUT_ROOT)

        # 总统计
        total_time = time.time() - start_time
        total_samples = sum(len(v) for v in all_results.values())

        print(f"\n{'=' * 80}")
        print(f"✅ 全部完成! 总样本: {total_samples} | 总耗时: {total_time:.1f}秒")
        print(f"   平均速度: {total_samples / total_time:.2f} 条/秒")
        print(f"   结果目录: {MultiCountryConfig.OUTPUT_ROOT}")
        print(f"{'=' * 80}")

    except KeyboardInterrupt:
        print("\n\n🛑 用户中断执行")
    except Exception as e:
        print(f"\n❌ 执行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await manager.close()


if __name__ == "__main__":
    # Windows事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 程序终止")
    except Exception as e:
        print(f"\n❌ 致命错误: {e}")
        import traceback

        traceback.print_exc()
