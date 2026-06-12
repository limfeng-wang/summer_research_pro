#!/usr/bin/env python3
"""
Step 3: 点击模拟 — SendInput 物理点击 + 人类时序，触发 XHS 浮层。

时序设计（模拟真实用户）:
  移动 → 停顿(80-200ms) → mousedown → 停顿(60-150ms) → mouseup → 等待浮层

验证:
  - 检测 __INITIAL_STATE__.note.noteDetailMap 是否有新数据
  - 检测 URL 是否变化（浮层或页面跳转）
"""

import ctypes
import json
import random
import sys
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xhs_snapshot import connect_chrome, extract_ssr_state, parse_note_detail, save_json, take_screenshot
from win32_api import (find_chrome_hwnd, force_foreground, move_mouse,
                       send_mousedown, send_mouseup,
                       get_cursor_pos as get_cursor, user32)

OUT_DIR = Path(__file__).resolve().parent / "输出数据" / f"step3_{datetime.now().strftime('%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    try:
        print(f"[{ts}] {msg}", flush=True)
    except Exception:
        pass




def calibrate(hwnd, page):
    """坐标校准（同 Step 2）。"""
    cr = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(cr))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))

    info = json.loads(page.run_js("""
        return JSON.stringify({
            innerWidth: window.innerWidth,
            innerHeight: window.innerHeight,
            dpr: window.devicePixelRatio,
        });
    """))
    return {
        'client_screen_x': pt.x,
        'client_screen_y': pt.y,
        'dpr': info['dpr'],
    }


def vp_to_screen(calib, vp_x, vp_y):
    return (
        int(calib['client_screen_x'] + vp_x * calib['dpr']),
        int(calib['client_screen_y'] + vp_y * calib['dpr']),
    )


def get_clickable_card(page):
    """找到一张完全可见、中心在视口内的卡片。"""
    raw = page.run_js("""
        const cards = [];
        document.querySelectorAll('a').forEach(a => {
            const href = a.getAttribute('href') || '';
            if (!href.includes('/explore/') && !href.includes('/search_result/')) return;
            const r = a.getBoundingClientRect();
            if (r.width < 100 || r.height < 100) return;
            const cy = r.y + r.height/2;
            if (cy < 100 || cy > window.innerHeight - 50) return;
            if (r.x + r.width/2 < 50 || r.x + r.width/2 > window.innerWidth - 50) return;
            const nm = href.match(/\\/(?:explore|search_result)\\/([a-f0-9]+)/);
            if (!nm) return;
            cards.push({
                note_id: nm[1],
                cx: Math.round(r.x + r.width/2),
                cy: Math.round(r.y + r.height/2),
                w: Math.round(r.width), h: Math.round(r.height),
                title: (a.textContent || '').trim().slice(0, 60),
                href: href.slice(0, 200),
            });
        });
        return JSON.stringify(cards);
    """)
    if raw:
        cards = json.loads(raw)
        if cards:
            # 选取中心最接近视口中央的卡片（最可能完全可见）
            best = min(cards, key=lambda c: abs(c['cy'] - 450))
            return best
    return None


# =============================================================================
# 检查浮层结果
# =============================================================================

def check_result(page, target_note_id):
    """检查点击是否触发了浮层/导航。"""
    current_url = page.url
    log(f"    当前 URL: {current_url[:120]}")

    # 检查 SSR
    ssr = extract_ssr_state(page)
    if not ssr:
        log("    无法提取 SSR")
        return None, None

    dm = ssr.get('note', {}).get('noteDetailMap') or {}
    log(f"    noteDetailMap 条目数: {len(dm)}, keys: {list(dm.keys())[:5]}")

    # 是否拿到目标笔记
    if target_note_id in dm:
        entry = dm[target_note_id]
        note = entry.get('note') if entry else None
        if note:
            log(f"    找到目标笔记!")
            return note, ssr

    # 是否有其他笔记数据（说明浮层打开了，但不是我们点的那个）
    for k, v in dm.items():
        note = v.get('note') if v else None
        if note and note.get('noteId') and note.get('title'):
            log(f"    发现其他笔记: key={k}, title={note.get('title', '')[:50]}")
            return note, ssr

    # 检查是否导航到了 explore 页面
    if '/explore/' in current_url:
        log("    检测到导航到 explore 页面")
        # 可能直接在页面上，尝试提取数据
        for k, v in dm.items():
            note = v.get('note') if v else None
            if note and note.get('noteId'):
                return note, ssr

    return None, ssr


# =============================================================================
# 主流程
# =============================================================================

def main():
    log("=" * 60)
    log("Step 3: 点击模拟 — SendInput 物理点击")
    log("=" * 60)

    # ---- 连接 Chrome ----
    log("\n[1] 连接 Chrome...")
    page = connect_chrome()
    current_url = page.url
    log(f"    当前 URL: {current_url[:120]}")

    # 如果不在搜索页，导航过去
    if '/search_result' not in current_url:
        kw = "牙痛"
        search_url = f"https://www.xiaohongshu.com/search_result?keyword={quote(kw)}"
        page.get(search_url, retry=2, timeout=25)
        try:
            page.wait.ele_displayed('a[href*="/explore/"], a.cover', timeout=20)
        except Exception:
            pass
        page.wait(1, 2)
    else:
        # 等页面稳定
        try:
            page.wait.ele_displayed('a[href*="/explore/"], a.cover', timeout=10)
        except Exception:
            pass
        page.wait(1, 2)

    log(f"    页面: {page.title[:60]}")

    # ---- 查找窗口 + 校准 + 前台 ----
    log("\n[2] 查找窗口 + 校准 + 前台...")
    wins = find_chrome_hwnd()
    if not wins:
        log("    未找到窗口")
        return page
    hwnd = wins[0]['hwnd']
    log(f"    HWND={hwnd} 位置({wins[0]['left']},{wins[0]['top']})")

    calib = calibrate(hwnd, page)
    log(f"    校准: client_screen=({calib['client_screen_x']},{calib['client_screen_y']}) dpr={calib['dpr']}")

    ok = force_foreground(hwnd)
    log(f"    前台化: {'OK' if ok else 'FAIL'}")
    time.sleep(0.3)

    # ---- 找到可点击的卡片 ----
    log("\n[3] 查找可点击卡片...")
    card = get_clickable_card(page)
    if not card:
        log("    未找到合适的卡片")
        return page

    note_id = card['note_id']
    log(f"    目标: note_id={note_id}")
    log(f"    标题: {card['title'][:60]}")
    log(f"    视口位置: ({card['cx']}, {card['cy']}) 尺寸: {card['w']}x{card['h']}")

    # 计算屏幕坐标
    sx, sy = vp_to_screen(calib, card['cx'], card['cy'])
    sx += random.randint(-3, 3)
    sy += random.randint(-3, 3)
    log(f"    屏幕坐标: ({sx}, {sy})")

    # ---- 前置截图 ----
    try:
        take_screenshot(page, OUT_DIR / "before_click.png")
    except Exception:
        pass

    # ---- 人类化移动 + 点击 ----
    log("\n[4] 执行人类化移动 + 点击...")

    # 移动到附近（模拟从不同方向接近）
    cur = get_cursor()
    approach_x = sx + random.randint(-120, -40)
    approach_y = sy + random.randint(-60, 60)
    approach_y = max(5, approach_y)

    log(f"    当前光标: {cur}  →  途径: ({approach_x}, {approach_y})  →  目标: ({sx}, {sy})")

    # 第一段：快速移到附近
    move_mouse(approach_x, approach_y)
    time.sleep(random.uniform(0.03, 0.08))

    # 第二段：慢速逼近目标（模拟精细调整）
    steps = random.randint(2, 4)
    for i in range(steps):
        frac = (i + 1) / steps
        ix = int(approach_x + (sx - approach_x) * frac + random.randint(-5, 5))
        iy = int(approach_y + (sy - approach_y) * frac + random.randint(-3, 3))
        move_mouse(ix, iy)
        time.sleep(random.uniform(0.04, 0.1))

    # 停顿（模拟阅读确认）
    dwell = random.uniform(0.1, 0.25)
    log(f"    到达目标，停顿 {dwell*1000:.0f}ms...")
    time.sleep(dwell)

    actual = get_cursor()
    log(f"    点击前光标: {actual}  (偏差: {actual[0]-sx}, {actual[1]-sy})")

    # ---- 点击！ ----
    log("\n[5] 点击！")
    send_mousedown()
    hold = random.uniform(0.07, 0.14)
    time.sleep(hold)
    send_mouseup()
    log(f"    mousedown → {hold*1000:.0f}ms → mouseup  完成")

    # ---- 等待浮层 ----
    log("\n[6] 等待浮层加载 (2.5-4s)...")
    time.sleep(random.uniform(2.5, 4.0))

    # ---- 检查结果 ----
    log("\n[7] 检查结果...")
    note_data, ssr = check_result(page, note_id)

    if note_data:
        parsed = parse_note_detail({'note': {'noteDetailMap': {note_id: {'note': note_data}}}})
        if parsed:
            save_json(OUT_DIR / "note_parsed.json", parsed)
            log(f"\n    *** SUCCESS! ***")
            log(f"    标题: {parsed.get('title', '')[:80]}")
            log(f"    点赞: {parsed['liked_count']}  收藏: {parsed['collected_count']}")
            log(f"    评论: {parsed['comment_count']}  分享: {parsed['share_count']}")
            log(f"    图片: {len(parsed.get('images', []))}张  标签: {len(parsed.get('hashtags', []))}个")
    else:
        log("\n    *** 浮层未触发 ***")
        if ssr:
            save_json(OUT_DIR / "ssr_after_click.json", ssr)

        # 诊断：检查是否有任何指针事件被触发
        diag = page.run_js(f"""
            const a = document.querySelector('a[href*="{note_id}"]');
            if (!a) return 'element not found';
            const r = a.getBoundingClientRect();
            const style = window.getComputedStyle(a);
            return JSON.stringify({{
                found: true,
                rect: {{x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)}},
                pointerEvents: style.pointerEvents,
                cursor: style.cursor,
                zIndex: style.zIndex,
                visibility: style.visibility,
                display: style.display,
                opacity: style.opacity,
            }});
        """)
        log(f"    元素诊断: {diag}")

    # ---- 后置截图 ----
    try:
        take_screenshot(page, OUT_DIR / "after_click.png")
    except Exception:
        pass

    log("\n" + "=" * 60)
    log("Step 3 完成 — 浏览器保持运行")
    log("=" * 60)

    return page


if __name__ == "__main__":
    main()
