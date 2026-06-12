#!/usr/bin/env python3
"""
fetch_policy.py — 决定哪些笔记需要打开详情页获取完整内容。

多级决策：
  1. 高互动（点赞+收藏+评论 > 阈值）→ 开（可能是优质内容）
  2. 标题含医学关键词 → 开（研究目标）
  3. 行为采样 → 开（即使不符合条件，保持行为自然）
  4. Session 上限 → 关（防止单次采集过多详情）
"""

import random

# 医学关键词（取决于研究主题，可扩展）
MEDICAL_KEYWORDS = ['牙', '痛', '炎', '肿', '疼', '智齿', '牙龈', '牙髓', '根管',
                    '口腔', '溃疡', '塞牙', '敏感', '蛀牙', '牙周', '拔牙']


class FetchPolicy:
    @staticmethod
    def should_fetch_detail(card: dict, detail_count: int = 0,
                            detail_limit: int = 30) -> bool:
        """
        card: FeedCardAdapter.parse() 输出的统一 dict
        detail_count: 当前 session 已开的详情数
        detail_limit: 单次 session 最大详情数
        """
        # --- Session 上限保护 ---
        if detail_count >= detail_limit:
            return False

        # --- 高互动触发 ---
        total_interaction = (card.get('liked_count', 0)
                            + card.get('collected_count', 0)
                            + card.get('comment_count', 0))
        if total_interaction > 100:
            return True

        # --- 医学关键词匹配 ---
        title = card.get('title', '') or ''
        if any(kw in title for kw in MEDICAL_KEYWORDS):
            return True

        # --- 行为采样：让"只看 feed 不开详情"的窗口里也穿插一些点击 ---
        if random.random() < 0.30:
            return True

        return False
