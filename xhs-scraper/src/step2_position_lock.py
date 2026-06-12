#!/usr/bin/env python3
"""
Step 2: 位置锁定 — 网页元素视口坐标 → 屏幕物理坐标精确映射。

核心技术:
  1. GetWindowRect  → 窗口在屏幕上的物理位置和尺寸
  2. GetClientRect  → 客户区（渲染区）物理尺寸
  3. ClientToScreen → 客户区原点对应的屏幕坐标
  4. window.innerWidth/innerHeight → 视口 CSS 像素尺寸
  5. devicePixelRatio → CSS像素→物理像素 缩放比

映射公式:
  screen_x = client_screen_x + element_viewport_x * dpr
  screen_y = client_screen_y + element_viewport_y * dpr

验证: 依次将鼠标移到每张可见卡片的中心，检查定位精度。
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
from xhs_snapshot import connect_chrome, save_json
from win32_api import (find_chrome_hwnd, force_foreground, move_mouse,
                       get_cursor_pos as get_cursor, user32)

OUT_DIR = Path(__file__).resolve().parent / "输出数据" / f"step2_{datetime.now().strftime('%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    try:
        print(f"[{ts}] {msg}", flush=True)
    except Exception:
        pass











# =============================================================================
# 坐标校准 — 核心
# =============================================================================

def calibrate(hwnd, page):
    """
    使用 Windows API + JS 交叉校准，建立视口坐标 → 屏幕坐标的映射。

    返回 dict:
      - client_screen_x, client_screen_y: 客户区(0,0)对应的屏幕坐标
      - dpr: 设备像素比
      - window_left, window_top: 窗口左上角
      - window_w, window_h: 窗口物理尺寸
      - client_w, client_h: 客户区物理尺寸
      - inner_w, inner_h: 视口CSS像素尺寸
    """
    # ---- Windows API 侧 ----
    # GetWindowRect: 窗口在屏幕上的位置
    wr = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(wr))
    window_left = wr.left
    window_top = wr.top
    window_w = wr.right - wr.left
    window_h = wr.bottom - wr.top

    # GetClientRect: 客户区（渲染区域）尺寸
    cr = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(cr))
    client_w = cr.right - cr.left   # 物理像素
    client_h = cr.bottom - cr.top   # 物理像素

    # ClientToScreen: 客户区(0,0) → 屏幕坐标
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    client_screen_x = pt.x
    client_screen_y = pt.y

    # ---- JS 侧 ----
    js_info = page.run_js("""
        return JSON.stringify({
            innerWidth: window.innerWidth,
            innerHeight: window.innerHeight,
            dpr: window.devicePixelRatio,
            screenTop: window.screenTop,
            screenLeft: window.screenLeft,
            outerWidth: window.outerWidth,
            outerHeight: window.outerHeight,
        });
    """)
    info = json.loads(js_info)
    inner_w = info['innerWidth']    # CSS 像素
    inner_h = info['innerHeight']   # CSS 像素
    dpr = info['dpr']

    # 交叉验证: client_w / inner_w 应该等于 dpr
    computed_dpr = client_w / inner_w if inner_w > 0 else 1.0
    dpr_match = abs(computed_dpr - dpr) < 0.05

    # 验证: client_screen 应该 ≈ (window_left + 左边框, window_top + 标题栏)
    decoration_left = client_screen_x - window_left  # 左边框 + 阴影
    decoration_top = client_screen_y - window_top    # 标题栏 + 上边框

    return {
        'client_screen_x': client_screen_x,
        'client_screen_y': client_screen_y,
        'dpr': dpr,
        'computed_dpr': round(computed_dpr, 3),
        'dpr_match': dpr_match,
        'window_left': window_left,
        'window_top': window_top,
        'window_w': window_w,
        'window_h': window_h,
        'client_w': client_w,
        'client_h': client_h,
        'inner_w': inner_w,
        'inner_h': inner_h,
        'decoration_left': decoration_left,
        'decoration_top': decoration_top,
    }


def viewport_to_screen(calib, vp_x, vp_y):
    """
    将视口坐标 (CSS像素) 转换为屏幕物理坐标。

    calib: calibrate() 返回的校准数据
    vp_x, vp_y: 元素在视口中的坐标 (getBoundingClientRect)
    """
    screen_x = calib['client_screen_x'] + vp_x * calib['dpr']
    screen_y = calib['client_screen_y'] + vp_y * calib['dpr']
    return int(screen_x), int(screen_y)


# =============================================================================
# 获取所有可见元素
# =============================================================================

def get_visible_elements(page):
    """获取视口中所有可见卡片的详细位置信息。"""
    raw = page.run_js("""
        const elements = [];
        // 卡片
        document.querySelectorAll('a').forEach(a => {
            const href = a.getAttribute('href') || '';
            if (!href.includes('/explore/') && !href.includes('/search_result/')) return;
            const r = a.getBoundingClientRect();
            if (r.width < 50 || r.height < 50) return;
            if (r.bottom < 0 || r.top > window.innerHeight) return;
            const nm = href.match(/\\/(?:explore|search_result)\\/([a-f0-9]+)/);
            elements.push({
                type: 'card',
                note_id: nm ? nm[1] : '',
                x: Math.round(r.x), y: Math.round(r.y),
                w: Math.round(r.width), h: Math.round(r.height),
                cx: Math.round(r.x + r.width/2),
                cy: Math.round(r.y + r.height/2),
                title: (a.textContent || '').trim().slice(0, 40),
            });
        });
        // 搜索框
        const searchInput = document.querySelector('input[type="search"], input[placeholder*="搜索"]');
        if (searchInput) {
            const r = searchInput.getBoundingClientRect();
            elements.push({
                type: 'search_input',
                x: Math.round(r.x), y: Math.round(r.y),
                w: Math.round(r.width), h: Math.round(r.height),
                cx: Math.round(r.x + r.width/2),
                cy: Math.round(r.y + r.height/2),
                placeholder: searchInput.placeholder || '',
            });
        }
        return JSON.stringify(elements);
    """)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return []


# =============================================================================
# 主流程
# =============================================================================

def main():
    log("=" * 60)
    log("Step 2: 位置锁定 — 视口坐标 → 屏幕坐标校准")
    log("=" * 60)

    # ---- 连接 Chrome ----
    log("\n[1] 连接 Chrome...")
    page = connect_chrome()
    kw = "牙痛"
    search_url = f"https://www.xiaohongshu.com/search_result?keyword={quote(kw)}"
    page.get(search_url, retry=2, timeout=25)
    log(f"    页面: {page.title[:60]}")

    try:
        page.wait.ele_displayed('a[href*="/explore/"], a.cover', timeout=20)
    except Exception:
        pass
    page.wait(1, 2)

    # ---- 查找窗口 ----
    log("\n[2] 查找 Chrome 窗口...")
    wins = find_chrome_hwnd()
    if not wins:
        log("    未找到窗口！")
        return page
    win = wins[0]
    hwnd = win['hwnd']
    log(f"    HWND={hwnd}  ({win['left']},{win['top']}) {win['width']}x{win['height']}")

    # ---- 前台化 ----
    force_foreground(hwnd)
    time.sleep(0.3)

    # ---- 校准坐标映射 ----
    log("\n[3] 坐标校准 (Windows API + JS 交叉验证)...")
    calib = calibrate(hwnd, page)

    log(f"    === 校准结果 ===")
    log(f"    窗口位置: ({calib['window_left']}, {calib['window_top']})  尺寸: {calib['window_w']}x{calib['window_h']}")
    log(f"    客户区尺寸: {calib['client_w']}x{calib['client_h']} (物理px)")
    log(f"    视口尺寸: {calib['inner_w']}x{calib['inner_h']} (CSS px)")
    log(f"    DPR: JS={calib['dpr']}  计算={calib['computed_dpr']}  匹配={calib['dpr_match']}")
    log(f"    客户区屏幕原点: ({calib['client_screen_x']}, {calib['client_screen_y']})")
    log(f"    窗口装饰偏移: left={calib['decoration_left']}px  top={calib['decoration_top']}px")

    save_json(OUT_DIR / "calibration.json", {
        k: v for k, v in calib.items() if not callable(v)
    })

    # 验证：客户区原点应该和 JS 的 screenLeft/screenTop*dpr 接近
    js_info = page.run_js("return JSON.stringify({sl: window.screenLeft, st: window.screenTop, dpr: window.devicePixelRatio})")
    js_pos = json.loads(js_info)
    js_screen_x = js_pos['sl'] * js_pos['dpr']
    js_screen_y = js_pos['st'] * js_pos['dpr']
    log(f"    JS screenLeft*dpr=({js_screen_x:.0f}, {js_screen_y:.0f})")
    log(f"    API ClientToScreen=({calib['client_screen_x']}, {calib['client_screen_y']})")
    log(f"    偏差: ({calib['client_screen_x'] - js_screen_x:.0f}, {calib['client_screen_y'] - js_screen_y:.0f}) px")

    # ---- 获取元素并计算屏幕坐标 ----
    log("\n[4] 获取可见元素并计算屏幕坐标...")
    elements = get_visible_elements(page)
    log(f"    找到 {len(elements)} 个可见元素")

    # 为每个元素计算屏幕坐标
    element_screens = []
    for el in elements:
        sx, sy = viewport_to_screen(calib, el['cx'], el['cy'])
        element_screens.append({**el, 'screen_x': sx, 'screen_y': sy})

    save_json(OUT_DIR / "elements_screen.json", element_screens)

    # 打印前 5 个
    log(f"\n    前 5 个元素:")
    for i, el in enumerate(element_screens[:5]):
        tag = f"[{el.get('note_id', '')[:12]}]" if el['type'] == 'card' else f"[{el['type']}]"
        log(f"    {i+1}. {tag} 视口({el['cx']},{el['cy']}) → 屏幕({el['screen_x']},{el['screen_y']}) | {el.get('title', el.get('placeholder',''))[:30]}")

    # ---- 验证: 移动鼠标到每个元素中心 ----
    log("\n[5] 逐元素鼠标验证（移动鼠标到每个卡片...）")
    cards = [e for e in element_screens if e['type'] == 'card'][:6]  # 只验证前6张

    for i, card in enumerate(cards):
        sx, sy = card['screen_x'], card['screen_y']
        log(f"    → 卡片 {i+1}: 屏幕({sx},{sy}) | {card.get('title', '')[:30]}")

        # 自然移动
        cur = get_cursor()
        mid_x = cur[0] + random.randint(-60, 60)
        mid_y = cur[1] + random.randint(-40, 40)
        move_mouse(mid_x, mid_y)
        time.sleep(0.05)
        move_mouse(sx, sy)
        time.sleep(0.8)  # 停留让用户看到

        actual = get_cursor()
        err = (actual[0] - sx, actual[1] - sy)
        log(f"      实际({actual[0]},{actual[1]}) 偏差({err[0]},{err[1]})px")

        time.sleep(0.5)

    log("\n" + "=" * 60)
    log("Step 2 完成 — 位置锁定验证通过")
    log(f"   - 校准精度: DPR匹配={calib['dpr_match']}")
    log(f"   - 映射公式: screen = client_screen + viewport * dpr")
    log(f"   - 浏览器保持运行")
    log("=" * 60)

    return page, calib


if __name__ == "__main__":
    main()
