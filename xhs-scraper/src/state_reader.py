#!/usr/bin/env python3
"""
state_reader.py — 从页面提取原始 __INITIAL_STATE__ 数据

策略优先级：
  1. SSR script 标签 → brace matching（最可靠，绕过 Vue reactive proxy）
  2. structuredClone 降级（部分 Vue proxy 可处理）

不依赖任何内部实现细节（_rawValue 等），只读公开的 SSR script 内容。
"""

import json
import re

# ---------------------------------------------------------------------------
# JS: Brace Matching — 从 <script> 标签中提取 window.__INITIAL_STATE__ = {...};
# 逐字符扫描，处理 JSON 字符串内的转义，找到匹配的闭合花括号。
# ---------------------------------------------------------------------------
SSR_EXTRACT_JS = """
    const scripts = document.querySelectorAll('script');
    for (let si = 0; si < scripts.length; si++) {
        const t = scripts[si].textContent || '';
        const idx = t.indexOf('window.__INITIAL_STATE__');
        if (idx === -1) continue;
        const start = t.indexOf('{', idx);
        if (start === -1) continue;
        let depth = 0, inStr = false, esc = false;
        for (let i = start; i < t.length; i++) {
            const c = t[i];
            if (esc) { esc = false; continue; }
            if (c === '\\\\' && inStr) { esc = true; continue; }
            if (c === '"') { inStr = !inStr; continue; }
            if (inStr) continue;
            if (c === '{') depth++;
            else if (c === '}') { depth--; if (depth === 0) return t.substring(start, i + 1); }
        }
    }
    return null;
"""

# ---------------------------------------------------------------------------
# JS: 从实时 Vue state 提取搜索结果卡片数据
# 依赖 Vue 3 ref 的内部结构（_rawValue），受版本耦合风险。
# 长期方案是用 network listener 替代。
# ---------------------------------------------------------------------------
SEARCH_FEEDS_JS = """
    var state = window.__INITIAL_STATE__;
    if (!state) return JSON.stringify([]);
    // 尝试用 structuredClone 避免 Vue proxy 干扰
    try { state = JSON.parse(JSON.stringify(state)); } catch(e) {}
    var items;
    // Path 1: search.noteItems (XHS 新版搜索数据路径, 2025+)
    if (Array.isArray(state.search?.noteItems)) items = state.search.noteItems;
    // Path 2: search.items (新版)
    if (!items && Array.isArray(state.search?.items)) items = state.search.items;
    // Path 3: search.feeds (旧版)
    if (!items && Array.isArray(state.search?.feeds)) items = state.search.feeds;
    // Path 4: search.notes (另一种可能路径)
    if (!items && Array.isArray(state.search?.notes)) items = state.search.notes;
    // Path 5: 顶层 items
    if (!items && Array.isArray(state.items)) items = state.items;
    // Path 6: note.noteDetailMap 取值（搜索结果页的另一种结构）
    if (!items && state.note?.noteDetailMap) {
        var dm = state.note.noteDetailMap;
        items = [];
        for (var k in dm) {
            if (dm[k] && dm[k].note) items.push(dm[k].note);
        }
    }
    if (!items || !items.length) return JSON.stringify([]);
    var results = [];
    for (var i = 0; i < items.length; i++) {
        var item = items[i];
        if (!item || !item.id && !item.noteId) continue;
        var card = item.noteCard || item.note || item;
        var user = card.user || {};
        var ii = card.interactInfo || {};
        results.push({
            note_id: item.id || item.noteId || '',
            xsec_token: item.xsecToken || card.xsecToken || '',
            title: card.displayTitle || card.title || '',
            type: card.type || '',
            author_name: user.nickname || user.nickName || '',
            author_id: user.userId || '',
            liked_count: +(ii.likedCount || ii.liked || 0),
            collected_count: +(ii.collectedCount || ii.collected || 0),
            comment_count: +(ii.commentCount || ii.comment || 0),
            share_count: +(ii.sharedCount || ii.shareCount || 0),
        });
    }
    return JSON.stringify(results);
"""

# ---------------------------------------------------------------------------
# JS Fallback: structuredClone（部分 Vue reactive proxy 可处理）
# ---------------------------------------------------------------------------
CLONE_STATE_JS = """
    try { return JSON.stringify(structuredClone(window.__INITIAL_STATE__)); }
    catch(e) { return null; }
"""

# ---------------------------------------------------------------------------
# JS: 诊断 — 输出 __INITIAL_STATE__ 的顶层 key 和 search 子结构
# ---------------------------------------------------------------------------
DIAGNOSE_STATE_KEYS_JS = """
    const state = window.__INITIAL_STATE__;
    if (!state) return JSON.stringify({_error: 'NO_INITIAL_STATE'});
    const info = {};
    const topKeys = Object.keys(state);
    info['_topKeys'] = topKeys;
    // 深搜所有 key 到 2 层
    for (const k of topKeys) {
        const v = state[k];
        if (v && typeof v === 'object' && !Array.isArray(v)) {
            info[k] = Object.keys(v);
            // 检查 search 子路径
            if (k === 'search') {
                for (const sk of Object.keys(v)) {
                    const sv = v[sk];
                    if (sv && typeof sv === 'object') {
                        info['search.' + sk] = Array.isArray(sv) ? 'array[' + sv.length + ']'
                            : (sv._rawValue ? 'VueRef(array[' + sv._rawValue.length + '])' : Object.keys(sv));
                    }
                }
            }
        } else if (Array.isArray(v)) {
            info[k] = 'array[' + v.length + ']';
        } else {
            info[k] = typeof v;
        }
    }
    return JSON.stringify(info);
"""


def _replace_undefined(text: str) -> str:
    """
    将 JS 文本中的 `undefined` 替换为 `null`，但跳过字符串内的出现，
    避免破坏字符串内容（如 `"reason: undefined"`）。
    """
    # 用正则分割字符串和非字符串片段
    parts = re.split(r'("(?:[^"\\]|\\.)*")', text)
    for i, part in enumerate(parts):
        if not part.startswith('"'):
            parts[i] = re.sub(r':undefined(?=\s*[,\]}])', ':null', part)
            parts[i] = re.sub(r'(?<=[,\[])\s*undefined(?=\s*[,\]}])', 'null', parts[i])
    return ''.join(parts)


def extract_ssr_state(page) -> dict | None:
    """
    从页面提取原始的 __INITIAL_STATE__ 字典。

    优先从 SSR <script> 标签中通过 brace matching 提取（避免 Vue reactive proxy 干扰）。
    失败时降级到 structuredClone。

    注意：搜索页的 feeds 数据不在 SSR 中（通过 XHR 后加载），需用
    extract_search_cards() 单独提取。
    """
    raw = page.run_js(SSR_EXTRACT_JS)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # XHS SSR JSON 用 undefined 替代 null
            # 注意：必须在非字符串上下文替换，避免破坏字符串内容
            cleaned = _replace_undefined(raw)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    raw = page.run_js(CLONE_STATE_JS)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    return None


def diagnose_state(page) -> dict | None:
    """
    诊断 __INITIAL_STATE__ 结构。返回顶层 key 和 search 子路径的概要。
    在 extract_search_cards 返回空时调用，确认数据存储位置。
    """
    raw = page.run_js(DIAGNOSE_STATE_KEYS_JS)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {'_error': 'json_decode_failed', '_raw': raw[:200]}
    return {'_error': 'js_returned_null'}


def extract_search_cards(page) -> list:
    """
    从搜索页实时 Vue state 提取卡片列表。

    依赖 Vue 3 ref._rawValue 获取原始数据（ref 内部实现），但有明确的
    版本耦合风险。XHS 搜索页的 feeds 数据通过 XHR 后加载写入 Vue state，
    不存在于 SSR script 标签中，因此必须从实时 state 读取。

    长期方案：用 network listener 拦截搜索 API 响应替代。
    """
    raw = page.run_js(SEARCH_FEEDS_JS)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return []


def extract_note_detail_from_state(state: dict) -> dict | None:
    """
    从详情页的 __INITIAL_STATE__ 中提取笔记完整信息。

    查找路径：state.note.noteDetailMap[id].note
    使用 currentNoteId / firstNoteId 优先匹配，无匹配则取第一个有效 entry。
    """
    if not state:
        return None
    note = state.get('note')
    if not note:
        return None
    dm = note.get('noteDetailMap') or {}
    if not dm:
        return None

    nid = note.get('currentNoteId') or note.get('firstNoteId')
    entry = None
    if nid and nid in dm:
        entry = dm[nid]
    else:
        for _k, v in dm.items():
            if v and v.get('note'):
                entry = v
                break
    if not entry:
        return None
    return entry.get('note')
