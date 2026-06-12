"""
agent_brain.py — AI 决策引擎

两种模式：
  LLM 模式（主力）：调用 OpenAI/Claude API，根据页面上下文、历史、进度做决策
  规则模式（Fallback）：无 API key 时的简单规则决策

用法:
    brain = AgentBrain()                     # 规则模式（无 API key）
    brain = AgentBrain(api_key="sk-...")     # LLM 模式
    action = brain.decide(keyword="牙疼", collected=5, target=100,
                          page_summary={...}, session_info={...})
    # → {"action": "SEARCH", "params": {"keyword": "牙痛"}, "reason": "..."}
"""

import json
import os
import random

# ---------------------------------------------------------------------------
# 可配置的关键词池 — 围绕主题的扩展搜索词
# ---------------------------------------------------------------------------
KEYWORD_VARIATIONS = {
    "牙疼": ["牙痛", "牙疼怎么办", "牙齿痛", "牙痛怎么缓解", "牙疼快速止痛",
             "牙龈肿痛", "牙神经痛", "牙齿敏感", "牙痛原因", "牙科"],
    "口腔溃疡": ["口腔溃疡怎么办", "反复口腔溃疡", "口腔溃疡快速愈合",
                 "口腔溃疡原因", "嘴巴溃疡"],
    "智齿": ["智齿发炎", "智齿冠周炎", "拔智齿", "智齿疼", "智齿肿痛"],
}

# 扩展关键词池 — 从各类牙科/口腔关键词衍生
FALLBACK_KEYWORDS = [
    "牙疼", "牙痛", "口腔溃疡", "智齿", "牙龈出血",
    "牙齿敏感", "蛀牙", "牙周炎", "根管治疗", "拔牙",
]

# ---------------------------------------------------------------------------
# 规则模式配置
# ---------------------------------------------------------------------------
RULE_CFG = {
    'open_threshold': 50,        # 互动量超过此值才值得打开
    'scroll_percent': 0.45,      # 滚动决策概率
    'new_keyword_percent': 0.25, # 换关键词概率
    'max_actions_before_stop': 3,
}


class AgentBrain:
    """决策引擎。根据采集进度和页面上下文决定下一步动作。"""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model

    # ======================================================================
    # 公开接口
    # ======================================================================

    def decide(self, keyword: str, collected: int, target: int,
               page_summary: dict, session_info: dict) -> dict:
        """返回动作字典：{"action": str, "params": dict, "reason": str}"""
        if self._api_key:
            # 直接尝试 LLM（不校验 key 格式，支持 OpenAI 兼容 API 代理 / Claude / 本地 LLM）
            return self._decide_llm(keyword, collected, target, page_summary, session_info)
        return self._decide_rules(keyword, collected, target, page_summary, session_info)

    # ======================================================================
    # LLM 模式
    # ======================================================================

    def _decide_llm(self, keyword, collected, target, page_summary, session_info) -> dict:
        prompt = self._build_prompt(keyword, collected, target, page_summary, session_info)
        system = (
            "你是一个模拟人类浏览小红书的行为决策器。你的任务是根据采集进度、"
            "会话历史和当前页面内容，决定下一步最自然的动作。\n"
            "规则：\n"
            "- 每次只做 1 步，模拟真人随便刷刷的感觉\n"
            "- 别盯着一个关键词反复搜，适时换词\n"
            "- 最多 3 步就 STOP\n"
            "- 不要贪多，更像是随便看看\n"
            "只输出 JSON：{\"action\": \"SEARCH|SCROLL|OPEN_NOTE|STOP\", "
            "\"params\": {...}, \"reason\": \"简短理由\"}"
        )
        try:
            text = self._call_llm(system, prompt)
            return self._parse_llm_response(text)
        except Exception as e:
            import sys as _sys
            print(f"[agent_brain] LLM 决策失败：{e}，降级到规则模式",
                  file=_sys.stderr)
            return self._decide_rules(keyword, collected, target, page_summary, session_info)

    def _call_llm(self, system: str, prompt: str) -> str:
        """调用 OpenAI API。"""
        from openai import OpenAI
        client = OpenAI(api_key=self._api_key)
        resp = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        return resp.choices[0].message.content or ""

    def _build_prompt(self, keyword, collected, target,
                      page_summary, session_info) -> str:
        visible = page_summary.get("visible_notes", [])
        lines = "\n".join(
            f"  {n['index']}. 「{n.get('title','?')}」"
            f" 👍{n.get('likes','?')} 💾{n.get('collects','?')}"
            for n in visible[:8]
        ) or "  （无可视笔记）"
        seen = page_summary.get("seen_ids", [])

        hours_ago = ""
        last_time = session_info.get("last_session_time")
        if last_time:
            hours_ago = f"（约 {session_info.get('hours_ago', '?')} 小时前）"

        return (
            f"收集目标：关键词「{keyword}」已收集 {collected}/{target} 篇\n"
            f"上次访问：{last_time or '无'}{hours_ago}\n"
            f"今日已访问：{session_info.get('today_count', 0)} 次\n"
            f"本会话已做：{session_info.get('actions_done', 0)} 步\n\n"
            f"当前页面：{page_summary.get('page_type', 'unknown')}\n"
            f"可见笔记：\n{lines}\n\n"
            f"已在本页看过的 note_id：{seen}\n\n"
            f"可选动作：\n"
            f"1. SEARCH(关键词) —— 搜一个新词\n"
            f"2. SCROLL —— 往下翻\n"
            f"3. OPEN_NOTE(序号) —— 点开第 N 篇笔记\n"
            f"4. STOP —— 结束本次会话\n\n"
            f"只输出 JSON。"
        )

    @staticmethod
    def _parse_llm_response(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
            text = text.strip()
        obj = json.loads(text)
        action = obj.get("action", "STOP")
        # 校验 action 合法性
        valid_actions = {"SEARCH", "SCROLL", "OPEN_NOTE", "STOP"}
        if action not in valid_actions:
            import sys as _sys
            print(f"[agent_brain] LLM 返回未知 action: {action}，降级为 STOP",
                  file=_sys.stderr)
            action = "STOP"
        params = obj.get("params", {})
        reason = obj.get("reason", "")
        return {"action": action, "params": params, "reason": reason}

    # ======================================================================
    # 规则模式（Fallback）
    # ======================================================================

    def _decide_rules(self, keyword, collected, target,
                      page_summary, session_info) -> dict:
        """基于规则的决策。不需要 API key。"""
        actions_done = session_info.get("actions_done", 0)

        # ① 已做够多 → STOP
        if actions_done >= RULE_CFG['max_actions_before_stop']:
            return {"action": "STOP", "params": {}, "reason": "本次已做够多了"}

        # ② 有未读笔记 → 筛选出标题与关键词相关的，再按互动量排序
        unread = [
            n for n in page_summary.get("visible_notes", [])
            if n.get("note_id") not in page_summary.get("seen_ids", [])
        ]
        # 先看标题是否相关
        relevant = [n for n in unread if self._is_relevant(n.get("title", ""), keyword)]
        if relevant:
            relevant.sort(key=lambda n: n.get("likes", 0) + n.get("collects", 0), reverse=True)
            best = relevant[0]
            total_int = best.get("likes", 0) + best.get("collects", 0)
            if total_int >= RULE_CFG['open_threshold'] or random.random() < 0.4:
                return {
                    "action": "OPEN_NOTE",
                    "params": {"index": best["index"]},
                    "reason": f"「{best.get('title','')[:20]}」相关，互动量 {total_int}",
                }

        # ③ 收集未满且动作数少 → 搜一个新词
        if collected < target and actions_done < 2:
            new_kw = self._pick_keyword(keyword)
            if new_kw:
                return {
                    "action": "SEARCH",
                    "params": {"keyword": new_kw},
                    "reason": f"换个词搜「{new_kw}」",
                }

        # ④ 随机概率滚动
        if random.random() < RULE_CFG['scroll_percent']:
            return {"action": "SCROLL", "params": {}, "reason": "随便翻翻"}

        # ⑤ 兜底
        return {"action": "STOP", "params": {}, "reason": "没有特别想做的"}

    @staticmethod
    def _is_relevant(title: str, keyword: str) -> bool:
        """判断笔记标题是否与搜索关键词相关。"""
        if not title:
            return False
        kw = keyword.lower()
        title_l = title.lower()
        # 标题直接包含关键词
        if kw in title_l:
            return True
        # 标题包含关键词的已知变体
        for var in KEYWORD_VARIATIONS.get(keyword, []):
            if var.lower() in title_l:
                return True
        # 标题包含通用牙科/口腔关键词
        DENTAL_TERMS = {'牙', '口腔', '齿', '龈', '髓', '根管', '溃疡', '塞牙'}
        for term in DENTAL_TERMS:
            if term in title:
                return True
        return False

    @staticmethod
    def _pick_keyword(current: str) -> str | None:
        """从变化词库选一个不同关键词。"""
        if current in KEYWORD_VARIATIONS:
            pool = KEYWORD_VARIATIONS[current]
        else:
            pool = [kw for kw in FALLBACK_KEYWORDS if kw != current]
        if not pool:
            return None
        return random.choice(pool)


# =============================================================================
# 自测
# =============================================================================
if __name__ == "__main__":
    brain = AgentBrain()  # 规则模式

    page = {
        "page_type": "search_result",
        "url": "https://xiaohongshu.com/search_result?keyword=牙疼",
        "keyword": "牙疼",
        "visible_notes": [
            {"index": 1, "title": "根管治疗后一定要戴牙冠吗？", "likes": 342, "collects": 891, "note_id": "a1"},
            {"index": 2, "title": "智齿发炎疼得睡不着", "likes": 128, "collects": 256, "note_id": "a2"},
            {"index": 3, "title": "牙龈出血怎么回事", "likes": 45, "collects": 67, "note_id": "a3"},
        ],
        "seen_ids": {"a1"},
    }
    info = {"last_session_time": "2026-05-13 08:00", "hours_ago": 6,
            "today_count": 2, "actions_done": 0}

    # 测试规则模式返回合法 action
    for step in range(5):
        act = brain.decide("牙疼", collected=34, target=100,
                           page_summary=page, session_info=info)
        assert act["action"] in ("SEARCH", "SCROLL", "OPEN_NOTE", "STOP"), f"bad action: {act}"
        info["actions_done"] += 1
        print(f"step {step}: {act['action']} — {act.get('reason', '')}")

    # 测试：动作数超限后应 STOP
    info["actions_done"] = 10
    act = brain.decide("牙疼", collected=50, target=100,
                       page_summary=page, session_info=info)
    assert act["action"] == "STOP", f"over limit should STOP, got {act}"

    # 测试：OPEN_NOTE 的 params 包含 index
    info["actions_done"] = 0
    act = brain.decide("牙疼", collected=5, target=100,
                       page_summary=page, session_info=info)
    if act["action"] == "OPEN_NOTE":
        assert "index" in act["params"], f"OPEN_NOTE missing index: {act}"

    print("agent_brain.py: all tests passed")
