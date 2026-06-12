"""
xhs_accumulator.py — 小红书长期数据积累微会话系统

「数据积累式采集」—— 极低频、长期、AI 驱动。

工作方式：
  1. 外部触发器（Task Scheduler / 手动）每小时调用一次
  2. 检查频率限制：距上次 ≥ 8h，日均 ≤ 3 次，半夜跳过
  3. 满足条件时执行一个「微会话」（1-3 步操作）
  4. 不满足时立即退出（< 1 秒）

用法:
    python xhs_accumulator.py --keyword "牙疼"
    python xhs_accumulator.py --keyword "牙疼" --target 100 --dry-run
    python xhs_accumulator.py --keyword "牙疼" --headless
"""

import argparse
import json
import random
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))

from DrissionPage import ChromiumPage, ChromiumOptions

from session_store import SessionStore
from agent_brain import AgentBrain
from state_reader import extract_ssr_state, extract_search_cards, extract_note_detail_from_state
from feed_adapter import DetailPageAdapter
from _common import (
    ANTI_DETECTION_JS, HUMAN_CFG, DELAY_CFG,
    human_dwell_time, random_scroll_amount,
)

# =============================================================================
# SQLite — notes 表（与 xhs_browse_collect.py 兼容）
# =============================================================================
NOTES_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword         TEXT,
    note_id         TEXT UNIQUE,
    title           TEXT,
    content         TEXT,
    author_name     TEXT,
    author_id       TEXT,
    liked_count     INTEGER DEFAULT 0,
    collected_count INTEGER DEFAULT 0,
    comment_count   INTEGER DEFAULT 0,
    share_count     INTEGER DEFAULT 0,
    publish_time    TEXT,
    images          TEXT,
    video_url       TEXT,
    hashtags        TEXT,
    source_url      TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# =============================================================================
# 工具函数
# =============================================================================

def _now_str() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(msg: str):
    try:
        print(f"[{_now_str()}] {msg}", flush=True)
    except UnicodeEncodeError:
        safe = msg.encode('utf-8', errors='replace').decode('utf-8')
        print(f"[{_now_str()}] {safe}", flush=True)


def _human_delay(action='reaction'):
    """使用 time.sleep 的延迟（不依赖 page 对象）"""
    low, high = DELAY_CFG.get(action, (0.3, 0.8))
    time.sleep(random.uniform(low, high))


# =============================================================================
# 浏览器管理
# =============================================================================

def _open_browser(profile_path: str, headless: bool = False):
    co = ChromiumOptions()
    co.set_argument("--no-sandbox")
    co.set_argument("--disable-blink-features=AutomationControlled")
    if headless:
        co.set_argument("--headless=new")
    co.set_argument(f"--user-data-dir={profile_path}")
    page = ChromiumPage(addr_or_opts=co)
    page.run_cdp('Page.addScriptToEvaluateOnNewDocument', source=ANTI_DETECTION_JS)
    return page


# =============================================================================
# 页面摘要 — 为 AI 决策提供上下文
# =============================================================================

def _extract_note_id_from_href(href: str) -> str:
    m = re.search(r'/(?:explore|search_result)/([a-f0-9]+)', href)
    if m:
        return m.group(1)
    m = re.search(r'/discovery/item/([a-f0-9]+)', href)
    return m.group(1) if m else ''


def _find_note_cards(page) -> list[dict]:
    """从页面中找到所有笔记卡片，返回 [{href, title, note_id}, ...]。"""
    try:
        raw = page.run_js("""
            const links = [];
            document.querySelectorAll('a').forEach(a => {
                const h = a.getAttribute('href') || '';
                if (h.includes('/explore/') || h.includes('/discovery/item/')) {
                    const img = a.querySelector('img');
                    links.push({
                        href: h,
                        title: (a.title || (img && img.alt) || '').substring(0, 80),
                    });
                }
            });
            return JSON.stringify(links);
        """)
        if raw:
            js_cards = json.loads(raw)
            result = []
            seen = set()
            for c in js_cards:
                note_id = _extract_note_id_from_href(c.get('href', ''))
                if note_id and note_id not in seen:
                    seen.add(note_id)
                    result.append({
                        'href': c['href'],
                        'title': c.get('title', '')[:60],
                        'note_id': note_id,
                    })
            if result:
                return result
    except Exception:
        pass

    try:
        cards = page.eles('a[href*="/explore/"]') or page.eles('a.cover')
        result = []
        for c in cards:
            href = c.attr('href') or ''
            note_id = _extract_note_id_from_href(href)
            if note_id:
                title = (c.attr('title') or '').strip()
                if not title:
                    img = c.ele('img')
                    title = (img.attr('alt') or '')[:60] if img else ''
                result.append({'href': href, 'title': title[:60], 'note_id': note_id})
        return result
    except Exception:
        pass

    return []


def _summarize_page(page, seen_ids: set) -> dict:
    url = page.url
    title = page.title

    if '/explore/' in url:
        page_type = 'note_detail'
    elif '/search_result' in url:
        page_type = 'search_result'
    elif '/discovery/' in url or '/feed/' in url:
        page_type = 'home'
    else:
        page_type = 'unknown'

    cards = _find_note_cards(page)

    interaction_lookup = {}
    if page_type == 'search_result':
        try:
            search_cards = extract_search_cards(page)
            for sc in search_cards:
                nid = sc.get('note_id', '')
                if nid:
                    interaction_lookup[nid] = {
                        'liked_count': sc.get('liked_count', 0),
                        'collected_count': sc.get('collected_count', 0),
                    }
        except Exception:
            pass

    visible_notes = []
    for idx, c in enumerate(cards[:15], start=1):
        nid = c['note_id']
        interactions = interaction_lookup.get(nid, {})
        visible_notes.append({
            'index': idx,
            'note_id': nid,
            'title': c.get('title', '')[:60],
            'likes': interactions.get('liked_count', 0),
            'collects': interactions.get('collected_count', 0),
        })

    return {
        'page_type': page_type,
        'url': url,
        'title': title,
        'visible_notes': visible_notes,
        'visible_count': len(visible_notes),
        'seen_ids': list(seen_ids),
    }


# =============================================================================
# 动作执行
# =============================================================================

def _do_search(page, keyword: str):
    search_url = f"https://www.xiaohongshu.com/search_result?keyword={quote(keyword)}"
    log(f"搜索：{keyword}")
    page.get(search_url, retry=2, timeout=25)
    _human_delay('read_normal')

    try:
        page.scroll.down(pixel=150)
        time.sleep(random.uniform(0.8, 1.5))
        page.scroll.down(pixel=250)
        time.sleep(random.uniform(1.0, 1.8))

        try:
            page.handle_alert(accept=True)
        except Exception:
            pass
    except Exception:
        pass

    log(f"搜索完成：{page.title} | {page.url}")


def _do_scroll(page):
    steps = random.randint(1, 3)
    total = 0
    for _ in range(steps):
        amount = random_scroll_amount()
        try:
            page.scroll.down(pixel=amount)
        except Exception:
            pass
        total += amount
        time.sleep(random.uniform(0.4, 1.0))
    log(f"向下滚动约 {total}px（{steps} 步）")


def _click_and_extract_detail(page, note_id: str, keyword: str,
                               store: SessionStore, conn: sqlite3.Connection) -> bool:
    log(f"点击卡片：{note_id}")

    try:
        el = page.ele(f'a[href*="{note_id}"]', timeout=5)
    except Exception:
        log(f"未找到 note_id {note_id} 的卡片")
        return False

    card_url = el.attr('href') or ''
    if not card_url:
        log("卡片元素无 href 属性")
        return False
    if card_url.startswith('/'):
        card_url = 'https://www.xiaohongshu.com' + card_url

    try:
        el.hover()
        time.sleep(random.uniform(0.1, 0.3))
    except Exception:
        pass

    original_tab_count = len(page.tab_ids)
    original_url = page.url
    original_tab_id = page.tab_id
    try:
        el.click()
    except Exception as e:
        log(f"点击异常：{e}")
        return False

    time.sleep(random.uniform(1.0, 2.5))

    try:
        page.handle_alert(accept=True)
    except Exception:
        pass

    detail_tab = None
    is_new_tab = False
    current_tabs = len(page.tab_ids)

    if current_tabs > original_tab_count:
        is_new_tab = True
        detail_tab = page.latest_tab
        try:
            detail_tab.run_cdp('Page.addScriptToEvaluateOnNewDocument',
                               source=ANTI_DETECTION_JS)
        except Exception:
            pass
        log(f"新标签页打开：{detail_tab.url}")
    elif page.url != original_url:
        detail_tab = page
        log(f"同标签页跳转：{page.url}")
    else:
        log("点击后未检测到导航")
        return False

    try:
        time.sleep(random.uniform(1.0, 2.0))
        dwell = human_dwell_time()
        log(f"停留 {dwell:.1f}s 阅读")

        if dwell > 5:
            if random.random() < HUMAN_CFG['detail_scroll_chance']:
                try:
                    detail_tab.scroll.down(pixel=random.randint(200, 600))
                except Exception:
                    pass
                time.sleep(random.uniform(0.3, 1.0))
            time.sleep(min(2.0, dwell * 0.3))
            remaining = dwell - 2.0
            if remaining > 0:
                time.sleep(remaining)
        else:
            time.sleep(dwell)

        state = extract_ssr_state(detail_tab)
        note_raw = extract_note_detail_from_state(state) if state else None
        detail = DetailPageAdapter.parse(note_raw) if note_raw else None

        if detail and detail.get('note_id'):
            conn.execute("""
                INSERT OR IGNORE INTO notes
                    (keyword, note_id, title, content, author_name, author_id,
                     liked_count, collected_count, comment_count, share_count,
                     publish_time, images, video_url, hashtags, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                keyword, detail['note_id'], detail.get('title', ''),
                detail.get('content', ''), detail.get('author_name', ''),
                detail.get('author_id', ''),
                detail.get('liked_count', 0), detail.get('collected_count', 0),
                detail.get('comment_count', 0), detail.get('share_count', 0),
                detail.get('publish_time', ''), detail.get('images', ''),
                detail.get('video_url', ''), detail.get('hashtags', ''),
                card_url,
            ))
            conn.commit()
            store.mark_note_collected(keyword, detail['note_id'])
            coll_count, _ = store.get_collection_progress(keyword)
            log(f"[OK] 已采集 {detail['note_id']}（{coll_count} 篇）")
            return True
        else:
            log(f"[SKIP] SSR 提取为空，跳过 {note_id}")

    except Exception as e:
        log(f"详情提取异常：{e}")

    finally:
        if is_new_tab and detail_tab:
            try:
                detail_tab.close()
            except Exception:
                pass
            try:
                page.activate_tab(original_tab_id)
            except Exception:
                pass
        else:
            try:
                page.back()
                time.sleep(random.uniform(1.0, 2.0))
            except Exception:
                pass

    return False


# =============================================================================
# 主流程
# =============================================================================

def run_micro_session(keyword: str, target: int = 100,
                      profile_path: str = '', headless: bool = False,
                      dry_run: bool = False) -> int:
    """执行一次微会话。返回本次采集的笔记数。"""
    store = SessionStore()

    # ========== ① 频率检查 ==========
    min_gap = 8.0      # 距上次至少 8 小时
    max_daily = 3      # 日均最多 3 次
    # 附加随机跳过（~30%，让行为更不规律）
    if random.random() < 0.30:
        log("随机跳过本次触发")
        return 0

    if not store.should_run_now(min_gap_hours=min_gap, max_daily=max_daily):
        log("频率限制跳过（时间未到 / 已达上限 / 睡眠时间）")
        return 0

    # ========== ② 打开浏览器 ==========
    log(f"打开浏览器（profile: {profile_path}, headless: {headless}）")
    if dry_run:
        log("DRY RUN — 跳过真实浏览器操作")
        return 0

    page = _open_browser(profile_path, headless)
    conn = sqlite3.connect(str(Path(__file__).resolve().parent / "输出数据" / "archive.db"))
    conn.execute(NOTES_SCHEMA)
    conn.commit()

    collected_ids = store.get_all_collected_note_ids()

    # ========== ③ 导航到搜索页 ==========
    _do_search(page, keyword)

    # ========== ④ AI 循环（最多 3 步） ==========
    session_id = store.log_session_start()
    actions_taken = 0
    notes_opened = 0
    current_keyword = keyword
    brain = AgentBrain()

    for step in range(3):
        page_summary = _summarize_page(page, collected_ids)

        collected, _ = store.get_collection_progress(current_keyword)
        session_info = {
            'actions_done': actions_taken,
            'today_count': store.get_today_session_count(),
            'last_session_time': (
                store.get_last_session_time().strftime('%Y-%m-%d %H:%M')
                if store.get_last_session_time() else None
            ),
        }
        if session_info['last_session_time']:
            try:
                last_dt = datetime.strptime(
                    session_info['last_session_time'], '%Y-%m-%d %H:%M')
                session_info['hours_ago'] = round(
                    (datetime.now() - last_dt).total_seconds() / 3600, 1)
            except Exception:
                pass

        decision = brain.decide(
            current_keyword, collected, target, page_summary, session_info)

        log(f"[step {step}] {decision['action']} — {decision.get('reason', '')}")

        if decision['action'] == 'SEARCH':
            new_kw = decision['params'].get('keyword', current_keyword)
            _do_search(page, new_kw)
            current_keyword = new_kw

        elif decision['action'] == 'SCROLL':
            _do_scroll(page)

        elif decision['action'] == 'OPEN_NOTE':
            idx = decision['params'].get('index', 1)
            try:
                cards = _find_note_cards(page)
                if not cards:
                    log("无可打开笔记，跳过 OPEN_NOTE")
                elif idx < 1 or idx > len(cards):
                    log(f"索引 {idx} 越界（共 {len(cards)} 张）")
                else:
                    card = cards[idx - 1]
                    href = card.get('href', '')
                    if href.startswith('/'):
                        href = 'https://www.xiaohongshu.com' + href
                    note_id = card.get('note_id', '')
                    if not note_id:
                        note_id = _extract_note_id_from_href(href)
                    if not note_id:
                        log("卡片 URL 无法提取 note_id")
                    elif note_id in collected_ids:
                        log(f"跳过已采集：{note_id}")
                    else:
                        success = _click_and_extract_detail(
                            page, note_id, current_keyword, store, conn)
                        if success:
                            notes_opened += 1
                            collected_ids.add(note_id)
                            coll, _ = store.get_collection_progress(current_keyword)
                            if coll >= target:
                                log(f"目标达成：{current_keyword} {coll}/{target}")
                                break
            except Exception as e:
                log(f"打开笔记异常：{e}")

        elif decision['action'] == 'STOP':
            log("AI 决定结束本次会话")
            break

        actions_taken += 1

        _human_delay('read_quick')

        if actions_taken >= 1 and random.random() < 0.15:
            log("随机提前结束会话")
            break

    # ========== ⑤ 清理 ==========
    page.quit()
    conn.close()
    store.log_session_end(
        session_id, actions=actions_taken, notes_opened=notes_opened,
        keyword=current_keyword,
    )
    log(f"微会话结束：{actions_taken} 步，{notes_opened} 篇新笔记")
    return notes_opened


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="小红书长期积累采集（微会话）")
    parser.add_argument("--keyword", default="牙疼", help="搜索关键词")
    parser.add_argument("--target", type=int, default=100, help="目标采集篇数")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--dry-run", action="store_true", help="只检查频率，不操作浏览器")
    parser.add_argument("--profile", default=None,
                        help="Chromium profile 路径（默认 ./chromium_profile）")
    args = parser.parse_args()

    profile = args.profile or str(
        Path(__file__).resolve().parent / "chromium_profile"
    )

    run_micro_session(
        keyword=args.keyword,
        target=args.target,
        profile_path=profile,
        headless=args.headless,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
