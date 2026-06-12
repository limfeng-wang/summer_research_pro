#!/usr/bin/env python3
"""
Step 3c: 事件对比 — 录制真实点击 vs 合成点击，找出差异。

注入完整事件记录器，然后等你用真实鼠标点击一次。
对比两种点击的事件属性差异。
"""

import ctypes, json, random, sys, time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

# 关键修复: 声明 DPI 感知，消除坐标虚拟化偏差
ctypes.windll.user32.SetProcessDPIAware()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xhs_snapshot import connect_chrome, extract_ssr_state, save_json, take_screenshot

OUT_DIR = Path(__file__).resolve().parent / "输出数据" / f"step3c_{datetime.now().strftime('%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    try: print(f"[{ts}] {msg}", flush=True)
    except: pass

def main():
    log("=" * 60)
    log("Step 3c: 真实点击 vs 合成点击 事件对比")
    log("=" * 60)

    page = connect_chrome()
    if '/search_result' not in page.url:
        kw = "牙痛"
        page.get(f"https://www.xiaohongshu.com/search_result?keyword={quote(kw)}", retry=2, timeout=25)
        try: page.wait.ele_displayed('a[href*="/explore/"], a.cover', timeout=20)
        except: pass
        page.wait(1, 2)
    log(f"页面: {page.title[:60]}")

    # ---- 注入超级事件记录器 ----
    log("\n[准备] 注入超级事件记录器...")
    page.run_js("""
        window.__real_events = [];
        window.__recording = false;

        function recordDetailed(e) {
            if (!window.__recording) return;
            // 只记录关键事件类型
            const keyTypes = ['pointerover', 'pointerenter', 'pointermove', 'pointerdown',
                             'pointerup', 'mouseover', 'mouseenter', 'mousemove',
                             'mousedown', 'mouseup', 'click'];
            if (!keyTypes.includes(e.type)) return;

            const rect = e.target ? (() => {
                try { const r = e.target.getBoundingClientRect(); return {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)}; }
                catch(ex) { return null; }
            })() : null;

            window.__real_events.push({
                type: e.type,
                time: Date.now(),
                isTrusted: e.isTrusted,
                clientX: Math.round(e.clientX),
                clientY: Math.round(e.clientY),
                screenX: Math.round(e.screenX),
                screenY: Math.round(e.screenY),
                target: (e.target.tagName || '') + (e.target.id ? '#'+e.target.id : '') + (e.target.className ? '.'+e.target.className.slice(0,50) : ''),
                targetRect: rect,
                button: e.button,
                buttons: e.buttons,
                detail: e.detail,
                // pointer 特有属性
                pointerType: e.pointerType,
                pointerId: e.pointerId,
                pressure: e.pressure,
                width: e.width,
                height: e.height,
                tiltX: e.tiltX,
                tiltY: e.tiltY,
                isPrimary: e.isPrimary,
            });
        }

        // 捕获阶段记录（确保我们拿到所有事件即使被 stopPropagation）
        document.addEventListener('pointerover', recordDetailed, true);
        document.addEventListener('pointerenter', recordDetailed, true);
        document.addEventListener('pointermove', recordDetailed, true);
        document.addEventListener('pointerdown', recordDetailed, true);
        document.addEventListener('pointerup', recordDetailed, true);
        document.addEventListener('mouseover', recordDetailed, true);
        document.addEventListener('mouseenter', recordDetailed, true);
        document.addEventListener('mousemove', recordDetailed, true);
        document.addEventListener('mousedown', recordDetailed, true);
        document.addEventListener('mouseup', recordDetailed, true);
        document.addEventListener('click', recordDetailed, true);

        'recorders injected'
    """)

    # ---- 提示用户点击 ----
    log("\n" + "!" * 60)
    log("!!! 请在 15 秒内用鼠标点击小红书搜索页上的任意一张卡片 !!!")
    log("!" * 60)

    # 开始录制
    page.run_js("window.__recording = true; window.__real_events = []; 'recording started'")
    log("录制开始...等待你的点击...")

    time.sleep(15)

    # 停止录制
    page.run_js("window.__recording = false; 'recording stopped'")
    real_events_raw = page.run_js("return JSON.stringify(window.__real_events)")
    real_events = json.loads(real_events_raw)
    log(f"录制到 {len(real_events)} 个真实事件")

    save_json(OUT_DIR / "real_human_click.json", real_events)

    # 分析真实点击的关键事件
    log("\n--- 真实点击事件序列分析 ---")
    down_idx = next((i for i, e in enumerate(real_events) if e['type'] in ('pointerdown', 'mousedown')), None)
    if down_idx is not None:
        # 显示 down 前 8 个事件 + down + up + click
        start = max(0, down_idx - 8)
        for i in range(start, min(len(real_events), down_idx + 5)):
            e = real_events[i]
            prefix = ">>>" if e['type'] in ('pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click') else "   "
            log(f"  {prefix} {e['type']:14s} client=({e['clientX']},{e['clientY']}) target={e['target'][:50]} pointerType={e.get('pointerType','')} pressure={e.get('pressure','')}")

    # 提取真实点击的 down 事件属性用于对比
    real_down = next((e for e in real_events if e['type'] == 'pointerdown'), None)
    real_up = next((e for e in real_events if e['type'] == 'pointerup'), None)

    if real_down:
        log(f"\n真实 pointerdown 关键属性:")
        for k in ['pointerType', 'pointerId', 'pressure', 'width', 'height', 'tiltX', 'tiltY', 'isPrimary', 'button', 'buttons', 'isTrusted']:
            log(f"  {k}: {real_down.get(k)}")

    # ---- 合成点击 (CDP) ----
    log("\n\n--- 合成点击事件 (CDP) ---")
    page.run_js("window.__recording = true; window.__real_events = []")

    # 先移动到卡片位置（用 JS 找到一张卡片）
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
        if (cards.length === 0) return null;
        return cards[0];
    """)

    if card_pos:
        cx, cy = card_pos['cx'], card_pos['cy']
        log(f"CDP 点击位置: ({cx}, {cy})")

        # 用 CDP 发送完整事件序列（模拟真实序列）
        # 1. mouseMoved (hover)
        page.run_cdp("Input.dispatchMouseEvent", type="mouseMoved", x=cx, y=cy)
        time.sleep(0.05)
        # 2. 微动
        page.run_cdp("Input.dispatchMouseEvent", type="mouseMoved", x=cx+random.randint(-2,2), y=cy+random.randint(-2,2))
        time.sleep(0.1)
        # 3. mousePressed
        page.run_cdp("Input.dispatchMouseEvent", type="mousePressed", x=cx, y=cy, button="left", clickCount=1)
        time.sleep(random.uniform(0.07, 0.13))
        # 4. mouseReleased
        page.run_cdp("Input.dispatchMouseEvent", type="mouseReleased", x=cx, y=cy, button="left", clickCount=1)
        time.sleep(2)

        page.run_js("window.__recording = false")
        synth_events = json.loads(page.run_js("return JSON.stringify(window.__real_events)"))
        log(f"CDP 合成产生 {len(synth_events)} 个事件")
        save_json(OUT_DIR / "cdp_synthetic_click.json", synth_events)

        for e in synth_events:
            log(f"     {e['type']:14s} trusted={e['isTrusted']} client=({e['clientX']},{e['clientY']}) target={e['target'][:50]}")

        # Check SSR
        ssr = extract_ssr_state(page)
        if ssr:
            dm = ssr.get('note', {}).get('noteDetailMap') or {}
            log(f"\nSSR noteDetailMap: {len(dm)} 条, keys: {list(dm.keys())[:5]}")
            for k, v in dm.items():
                note = v.get('note') if v else None
                if note and note.get('title'):
                    log(f"  → {k}: {note.get('title','')[:60]}")

        # ---- 对比 ----
        log(f"\n{'='*60}")
        log("事件对比完成")
        log(f"  真实点击事件数: {len(real_events)}")
        log(f"  CDP合成事件数: {len(synth_events)}")
        if real_down:
            log(f"  真实 pointerdown: pointerType={real_down.get('pointerType')}, pressure={real_down.get('pressure')}")
        synth_down = next((e for e in synth_events if e['type']=='pointerdown'), None)
        if synth_down:
            log(f"  CDP pointerdown: pointerType={synth_down.get('pointerType')}, pressure={synth_down.get('pressure')}")

    log("\n浏览器保持运行，数据已保存")
    return page

if __name__ == "__main__":
    main()
