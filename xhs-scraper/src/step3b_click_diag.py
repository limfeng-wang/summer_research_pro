#!/usr/bin/env python3
"""
Step 3b: 点击诊断 — 注入全局事件监听，验证 SendInput 点击是否到达页面。
"""

import ctypes, json, random, sys, time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xhs_snapshot import connect_chrome, extract_ssr_state, save_json, take_screenshot
from win32_api import (find_chrome_hwnd, force_foreground, move_mouse,
                       send_mousedown, send_mouseup,
                       get_cursor_pos as get_cursor, user32)

OUT_DIR = Path(__file__).resolve().parent / "输出数据" / f"step3b_{datetime.now().strftime('%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    try: print(f"[{ts}] {msg}", flush=True)
    except: pass

def calibrate(hwnd, page):
    cr = wintypes.RECT(); user32.GetClientRect(hwnd, ctypes.byref(cr))
    pt = wintypes.POINT(0, 0); user32.ClientToScreen(hwnd, ctypes.byref(pt))
    info = json.loads(page.run_js("return JSON.stringify({iw: window.innerWidth, ih: window.innerHeight, dpr: window.devicePixelRatio})"))
    return {'csx': pt.x, 'csy': pt.y, 'dpr': info['dpr']}

def vp2screen(calib, vx, vy):
    return (int(calib['csx'] + vx * calib['dpr']), int(calib['csy'] + vy * calib['dpr']))


def main():
    log("=" * 60)
    log("Step 3b: 点击诊断 — 全局事件监听")
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
    hwnd = wins[0]['hwnd']
    calib = calibrate(hwnd, page)
    log(f"校准: cs=({calib['csx']},{calib['csy']}) dpr={calib['dpr']}")
    force_foreground(hwnd)

    # ---- 注入全局事件监听器 ----
    log("\n[关键] 注入全局事件监听器...")
    page.run_js("""
        window.__click_diag = [];
        window.__last_event = null;

        function recordEvent(e) {
            window.__last_event = {
                type: e.type,
                isTrusted: e.isTrusted,
                clientX: Math.round(e.clientX),
                clientY: Math.round(e.clientY),
                screenX: Math.round(e.screenX),
                screenY: Math.round(e.screenY),
                target: (e.target.tagName || '') + (e.target.className ? '.' + e.target.className.slice(0,40) : ''),
                targetHref: (e.target.href || '').slice(0,100),
                time: Date.now(),
            };
            window.__click_diag.push(window.__last_event);
            console.log('[DIAG]', e.type, e.isTrusted, Math.round(e.clientX), Math.round(e.clientY), window.__last_event.target);
        }

        document.addEventListener('mousedown', recordEvent, true);
        document.addEventListener('mouseup', recordEvent, true);
        document.addEventListener('click', recordEvent, true);
        document.addEventListener('pointerdown', recordEvent, true);
        document.addEventListener('pointerup', recordEvent, true);
        'listeners injected'
    """)
    log("    监听器已注入 (捕获阶段)")

    # ---- 找到一张卡片 ----
    card_data = page.run_js("""
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
                cx: Math.round(r.x + r.width/2), cy: Math.round(r.y + r.height/2),
                w: Math.round(r.width), h: Math.round(r.height),
                x: Math.round(r.x), y: Math.round(r.y),
                title: (a.textContent || '').trim().slice(0, 50),
            });
        });
        return JSON.stringify(cards.length > 0 ? cards : []);
    """)
    cards = json.loads(card_data)
    if not cards:
        log("无可用卡片")
        return page

    # 选视口中央的卡片
    best = min(cards, key=lambda c: abs(c['cy'] - 450))
    note_id = best['note_id']
    vp_cx, vp_cy = best['cx'], best['cy']
    sx, sy = vp2screen(calib, vp_cx, vp_cy)
    log(f"目标: {note_id}  视口({vp_cx},{vp_cy}) 尺寸{best['w']}x{best['h']}  屏幕({sx},{sy})")
    log(f"卡片区域: viewport ({best['x']},{best['y']}) - ({best['x']+best['w']},{best['y']+best['h']})")

    # ---- 先做 CDP 点击测试 (验证事件是否被接收) ----
    log("\n[测试A] CDP Input.dispatchMouseEvent 点击（isTrusted=false）...")
    page.run_cdp("Input.dispatchMouseEvent", type="mouseMoved",
                 x=vp_cx, y=vp_cy)
    time.sleep(0.05)
    page.run_cdp("Input.dispatchMouseEvent", type="mousePressed",
                 x=vp_cx, y=vp_cy, button="left", clickCount=1)
    time.sleep(0.1)
    page.run_cdp("Input.dispatchMouseEvent", type="mouseReleased",
                 x=vp_cx, y=vp_cy, button="left", clickCount=1)
    time.sleep(1)

    cdp_events = page.run_js("return JSON.stringify(window.__click_diag)")
    cdp_diag = json.loads(cdp_events)
    log(f"    CDP点击后记录 {len(cdp_diag)} 个事件:")
    for e in cdp_diag:
        log(f"      {e['type']:12s} trusted={e['isTrusted']}  client=({e['clientX']},{e['clientY']})  target={e['target'][:60]}")

    # 清空记录
    page.run_js("window.__click_diag = []")

    # ---- SendInput 物理点击 ----
    log(f"\n[测试B] SendInput 物理点击（isTrusted=true）...")

    # 移动鼠标
    cur = get_cursor()
    move_mouse(sx + random.randint(-80, -30), max(5, sy + random.randint(-40, 40)))
    time.sleep(0.03)
    move_mouse(sx, sy)
    time.sleep(random.uniform(0.15, 0.3))
    actual = get_cursor()
    log(f"    光标: ({actual[0]},{actual[1]}) 目标:({sx},{sy})")

    # 点击
    send_mousedown()
    time.sleep(random.uniform(0.08, 0.14))
    send_mouseup()
    log("    物理点击完成")

    time.sleep(1.5)

    # 读取事件记录
    phys_events = page.run_js("return JSON.stringify(window.__click_diag)")
    phys_diag = json.loads(phys_events)
    log(f"    SendInput点击后记录 {len(phys_diag)} 个事件:")
    for e in phys_diag:
        log(f"      {e['type']:12s} trusted={e['isTrusted']}  client=({e['clientX']},{e['clientY']})  screen=({e['screenX']},{e['screenY']})  target={e['target'][:60]}")

    if len(phys_diag) == 0:
        log("    *** 页面未收到任何鼠标事件！***")
        log("    这意味着 SendInput 事件没有到达 Chrome 的渲染进程")

    # ---- 检查 SSR ----
    log(f"\n[SSR检查]")
    ssr = extract_ssr_state(page)
    if ssr:
        dm = ssr.get('note', {}).get('noteDetailMap') or {}
        log(f"    noteDetailMap: {len(dm)} 条, keys: {list(dm.keys())[:5]}")
        for k, v in dm.items():
            note = v.get('note') if v else None
            if note and note.get('title'):
                log(f"    → {k}: {note.get('title','')[:60]}")
    else:
        log("    无法提取SSR")

    # 保存
    save_json(OUT_DIR / "cdp_events.json", cdp_diag)
    save_json(OUT_DIR / "sendinput_events.json", phys_diag)
    try: take_screenshot(page, OUT_DIR / "screenshot.png")
    except: pass

    log("\n浏览器保持运行")
    return page

if __name__ == "__main__":
    main()
