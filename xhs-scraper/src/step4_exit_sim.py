#!/usr/bin/env python3
"""
Step 4: 退出模拟 — 打开浮层 → 检测浮层 → 点击遮罩关闭 → 回到搜索页。

真实用户行为:
  点击卡片 → 浮层打开 → 阅读 → 点击左侧暗色遮罩区域 → 浮层关闭 → 回到搜索页

技术实现:
  1. CDP Input.dispatchMouseEvent 打开浮层
  2. DOM 检测浮层是否打开 (note-detail-mask)
  3. 获取遮罩元素坐标
  4. SendInput 物理点击遮罩空白区域
  5. 检测浮层关闭
"""

import ctypes, json, random, sys, time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

ctypes.windll.user32.SetProcessDPIAware()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xhs_snapshot import connect_chrome, extract_ssr_state, save_json, take_screenshot
from win32_api import (find_chrome_hwnd, force_foreground, move_mouse,
                       send_mousedown, send_mouseup,
                       get_cursor_pos as get_cursor, user32)

OUT_DIR = Path(__file__).resolve().parent / "输出数据" / f"step4_{datetime.now().strftime('%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    try: print(f"[{ts}] {msg}", flush=True)
    except: pass

def calibrate(hwnd, page):
    cr = wintypes.RECT(); user32.GetClientRect(hwnd, ctypes.byref(cr))
    pt = wintypes.POINT(0, 0); user32.ClientToScreen(hwnd, ctypes.byref(pt))
    info = json.loads(page.run_js("return JSON.stringify({dpr: window.devicePixelRatio})"))
    return {'csx': pt.x, 'csy': pt.y, 'dpr': info['dpr']}

def vp2screen(calib, vx, vy):
    return (int(calib['csx'] + vx * calib['dpr']), int(calib['csy'] + vy * calib['dpr']))


# =============================================================================
# 浮层检测
# =============================================================================

def is_overlay_open(page):
    """检测浮层是否已打开。"""
    result = page.run_js("""
        const mask = document.querySelector('.note-detail-mask');
        const container = document.querySelector('#noteContainer');
        if (!mask || !container) return JSON.stringify({open: false});
        const mr = mask.getBoundingClientRect();
        const cr = container.getBoundingClientRect();
        return JSON.stringify({
            open: true,
            mask: {x: Math.round(mr.x), y: Math.round(mr.y), w: Math.round(mr.width), h: Math.round(mr.height)},
            container: {x: Math.round(cr.x), y: Math.round(cr.y), w: Math.round(cr.width), h: Math.round(cr.height)},
        });
    """)
    if result:
        data = json.loads(result)
        return data
    return {'open': False}


# =============================================================================
# 主流程
# =============================================================================

def main():
    log("=" * 60)
    log("Step 4: 退出模拟 — 点击遮罩关闭浮层")
    log("=" * 60)

    page = connect_chrome()
    if '/search_result' not in page.url:
        kw = "牙痛"
        page.get(f"https://www.xiaohongshu.com/search_result?keyword={quote(kw)}", retry=2, timeout=25)
        try: page.wait.ele_displayed('a[href*="/explore/"], a.cover', timeout=20)
        except: pass
        page.wait(1, 2)
    log(f"页面: {page.title[:60]}")

    wins = find_chrome_hwnd()
    hwnd = wins[0]['hwnd'] if wins else None
    calib = calibrate(hwnd, page) if hwnd else {'csx': 0, 'csy': 0, 'dpr': 1}
    if hwnd:
        log(f"窗口: ({wins[0]['left']},{wins[0]['top']}) dpr={calib['dpr']}")
        force_foreground(hwnd)

    # ---- 找卡片 ----
    card_pos = page.run_js("""
        const cards = [];
        document.querySelectorAll('a').forEach(a => {
            const href = a.getAttribute('href') || '';
            if (!href.includes('/explore/') && !href.includes('/search_result/')) return;
            const r = a.getBoundingClientRect();
            if (r.width < 100 || r.height < 100) return;
            if (r.y + r.height/2 < 100 || r.y + r.height/2 > window.innerHeight - 50) return;
            cards.push({cx: Math.round(r.x + r.width/2), cy: Math.round(r.y + r.height/2)});
        });
        return cards.length > 0 ? JSON.stringify(cards[0]) : null;
    """)
    if not card_pos:
        log("无可用卡片")
        return page
    if isinstance(card_pos, str):
        card_pos = json.loads(card_pos)
    cx, cy = card_pos['cx'], card_pos['cy']
    log(f"卡片视口: ({cx}, {cy})")

    # ---- Step A: 打开浮层 (CDP) ----
    log("\n[A] 打开浮层...")
    page.run_cdp("Input.dispatchMouseEvent", type="mouseMoved", x=cx, y=cy)
    time.sleep(0.08)
    page.run_cdp("Input.dispatchMouseEvent", type="mousePressed", x=cx, y=cy, button="left", clickCount=1)
    time.sleep(random.uniform(0.07, 0.13))
    page.run_cdp("Input.dispatchMouseEvent", type="mouseReleased", x=cx, y=cy, button="left", clickCount=1)
    time.sleep(2)

    overlay = is_overlay_open(page)
    log(f"浮层状态: {'已打开' if overlay['open'] else '未打开'}")
    if overlay['open']:
        log(f"  遮罩: {overlay['mask']}")
        log(f"  容器: {overlay['container']}")
    save_json(OUT_DIR / "overlay_state.json", overlay)

    try: take_screenshot(page, OUT_DIR / "overlay_open.png")
    except: pass

    if not overlay['open']:
        log("浮层未打开，无法测试退出")
        return page

    # ---- Step B: 找遮罩可点击区域 ----
    log("\n[B] 计算遮罩点击位置...")
    # 容器在右侧，遮罩覆盖全屏。安全点击位置：遮罩左侧区域，避开容器
    container = overlay['container']
    mask = overlay['mask']

    # 在容器左侧 100px（确保在遮罩上，不在容器内）
    dismiss_vp_x = max(50, container['x'] - 80)
    dismiss_vp_y = container['y'] + container['h'] // 2

    log(f"  容器区域: x={container['x']}-{container['x']+container['w']}, y={container['y']}-{container['y']+container['h']}")
    log(f"  遮罩点击: 视口({dismiss_vp_x}, {dismiss_vp_y})")

    # ---- Step C: 物理点击遮罩 (SendInput) ----
    log("\n[C] SendInput 点击遮罩...")
    if hwnd:
        sx, sy = vp2screen(calib, dismiss_vp_x, dismiss_vp_y)
        log(f"  屏幕坐标: ({sx}, {sy})")

        # 确保前台
        force_foreground(hwnd)
        time.sleep(0.2)

        # 移动 + 点击
        cur = get_cursor()
        move_mouse(sx + random.randint(-40, -10), max(5, sy + random.randint(-30, 30)))
        time.sleep(0.04)
        move_mouse(sx, sy)
        time.sleep(random.uniform(0.1, 0.2))

        send_mousedown()
        time.sleep(random.uniform(0.06, 0.12))
        send_mouseup()
        log("  SendInput 点击完成")
    else:
        log("  无窗口句柄，改用 CDP 点击")
        page.run_cdp("Input.dispatchMouseEvent", type="mouseMoved", x=dismiss_vp_x, y=dismiss_vp_y)
        time.sleep(0.05)
        page.run_cdp("Input.dispatchMouseEvent", type="mousePressed", x=dismiss_vp_x, y=dismiss_vp_y, button="left", clickCount=1)
        time.sleep(0.08)
        page.run_cdp("Input.dispatchMouseEvent", type="mouseReleased", x=dismiss_vp_x, y=dismiss_vp_y, button="left", clickCount=1)

    time.sleep(1.5)

    # ---- Step D: 验证浮层关闭 ----
    log("\n[D] 验证浮层关闭...")
    overlay2 = is_overlay_open(page)
    log(f"浮层状态: {'仍打开' if overlay2['open'] else '已关闭'}")

    # 检查 URL 是否还在搜索页（没有导航走）
    current_url = page.url
    log(f"当前 URL: {current_url[:120]}")
    log(f"URL 仍在搜索页: {'/search_result' in current_url}")

    # 检查搜索页卡片还在不在
    cards_after = page.run_js("""
        const count = document.querySelectorAll('a[href*="/explore/"], a[href*="/search_result/"]').length;
        return JSON.stringify({cardLinks: count});
    """)
    log(f"搜索页卡片链接数: {cards_after}")

    save_json(OUT_DIR / "after_dismiss.json", {
        'overlay_closed': not overlay2['open'],
        'url_on_search': '/search_result' in current_url,
        'cards_after': cards_after,
    })

    try: take_screenshot(page, OUT_DIR / "overlay_closed.png")
    except: pass

    # ---- 总结 ----
    log(f"\n{'='*60}")
    log("Step 4 完成 — 退出模拟")
    log(f"  浮层打开: OK")
    log(f"  遮罩点击: OK")
    log(f"  浮层关闭: {'OK' if not overlay2['open'] else 'FAIL'}")
    log(f"  回到搜索页: {'OK' if '/search_result' in current_url else 'FAIL'}")
    log(f"  浏览器保持运行")
    log("=" * 60)

    return page


if __name__ == "__main__":
    main()
