#!/usr/bin/env python3
"""
xhs_collector_v2.py — 精简版采集器

核心原则:
  - 只保留已验证可行的逻辑
  - 不预先滚动
  - 点击 → 检查浮层 → 提取 → 退出 → 下一张
  - 只有可视卡片用尽时才滚动
"""

import ctypes, json, random, sqlite3, sys, time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

ctypes.windll.user32.SetProcessDPIAware()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xhs_snapshot import (
    connect_chrome, log, save_json, take_screenshot,
)
from win32_api import (
    find_chrome_hwnd, force_foreground, move_mouse, send_click,
    send_mousedown, send_mouseup, get_cursor_pos, calibrate, vp2screen,
    user32, is_windows,
)

OUT_ROOT = Path(__file__).resolve().parent / "输出数据"

# =============================================================================
# 卡片操作 (精简版 — 只保留已验证的逻辑)
# =============================================================================

def get_cards(page):
    """获取可见卡片，与 step3b 完全相同的逻辑。"""
    raw = page.run_js("""
        const cards = [];
        const seen = new Set();
        document.querySelectorAll('a').forEach(a => {
            if (a.closest('.note-detail-mask, #noteContainer, [class*="note-detail"]')) return;
            const href = a.getAttribute('href') || '';
            if (!href.includes('/explore/') && !href.includes('/search_result/')) return;
            const r = a.getBoundingClientRect();
            if (r.width < 100 || r.height < 100) return;
            const cy = r.y + r.height/2;
            if (cy < 100 || cy > window.innerHeight - 50) return;
            if (r.x + r.width/2 < 50 || r.x + r.width/2 > window.innerWidth - 50) return;
            const nm = href.match(/\\/(?:explore|search_result)\\/([a-f0-9]+)/);
            if (!nm) return;
            const nid = nm[1];
            if (seen.has(nid)) return;  // 同一 note_id 可能多个 <a> 标签（卡片+推荐位）
            seen.add(nid);
            cards.push({
                note_id: nid,
                cx: Math.round(r.x + r.width/2), cy: Math.round(r.y + r.height/2),
                w: Math.round(r.width), h: Math.round(r.height),
                x: Math.round(r.x), y: Math.round(r.y),
                title: (a.textContent || '').trim().slice(0, 60),
            });
        });
        return JSON.stringify(cards);
    """)
    if raw:
        try:
            cards = json.loads(raw)
            # 去重：同一 note_id 可能出现在多个 <a> 标签中
            seen = set()
            unique = []
            for c in cards:
                if c['note_id'] not in seen:
                    seen.add(c['note_id'])
                    unique.append(c)
            return unique
        except: pass
    return []

def is_overlay_open(page):
    result = page.run_js("""
        const mask = document.querySelector('.note-detail-mask');
        const container = document.querySelector('#noteContainer');
        if (!mask || !container) return JSON.stringify({open: false});
        const cr = container.getBoundingClientRect();
        return JSON.stringify({
            open: true,
            container: {x: Math.round(cr.x), y: Math.round(cr.y), w: Math.round(cr.width), h: Math.round(cr.height)},
        });
    """)
    if result:
        try: return json.loads(result)
        except: pass
    return {'open': False}

def open_card(page, card):
    """CDP 点击卡片打开浮层。"""
    cx, cy = card['cx'], card['cy']
    try:
        page.run_cdp("Input.dispatchMouseEvent", type="mouseMoved", x=cx - 30, y=max(10, cy - 30))
        time.sleep(0.04)
        page.run_cdp("Input.dispatchMouseEvent", type="mouseMoved", x=cx, y=cy)
        time.sleep(random.uniform(0.06, 0.15))
        page.run_cdp("Input.dispatchMouseEvent", type="mousePressed", x=cx, y=cy, button="left", clickCount=1)
        time.sleep(random.uniform(0.07, 0.14))
        page.run_cdp("Input.dispatchMouseEvent", type="mouseReleased", x=cx, y=cy, button="left", clickCount=1)
        time.sleep(random.uniform(1.8, 3.0))
        return True
    except Exception as e:
        log(f"  CDP错误: {e}")
        return False

def close_overlay(page, calib):
    """多策略关闭浮层，依次尝试 ESC → CDP 点击遮罩 → SendInput。"""
    if not is_overlay_open(page)['open']:
        return True
    c = is_overlay_open(page)['container']

    # Strategy 1: ESC 键 (CDP dispatchKeyEvent, 比 JS dispatchEvent 更可靠)
    try:
        page.run_cdp("Input.dispatchKeyEvent", type="rawKeyDown", key="Escape",
                     windowsVirtualKeyCode=27, code="Escape")
        time.sleep(0.1)
        page.run_cdp("Input.dispatchKeyEvent", type="keyUp", key="Escape",
                     windowsVirtualKeyCode=27, code="Escape")
        time.sleep(0.8)
        if not is_overlay_open(page)['open']:
            return True
    except Exception as e:
        log(f"  ESC 关闭失败 (CDP): {e}")

    # Strategy 2: CDP 点击遮罩区域
    dx = max(50, c['x'] - 80)
    dy = c['y'] + c['h'] // 2
    try:
        page.run_cdp("Input.dispatchMouseEvent", type="mouseMoved", x=dx, y=dy)
        time.sleep(0.05)
        page.run_cdp("Input.dispatchMouseEvent", type="mousePressed", x=dx, y=dy, button="left", clickCount=1)
        time.sleep(0.08)
        page.run_cdp("Input.dispatchMouseEvent", type="mouseReleased", x=dx, y=dy, button="left", clickCount=1)
        time.sleep(1.0)
        if not is_overlay_open(page)['open']:
            return True
    except Exception as e:
        log(f"  CDP 点击关闭失败: {e}")

    # Strategy 3: SendInput 物理点击 (仅 Windows)
    if calib.get('hwnd'):
        try:
            sx, sy = vp2screen(calib, dx, dy)
            force_foreground(calib['hwnd'])
            time.sleep(0.1)
            move_mouse(sx, sy)
            time.sleep(random.uniform(0.08, 0.15))
            # 校验鼠标是否在目标位置（防用户干扰）
            cx, cy = get_cursor_pos()
            if abs(cx - sx) > 10 or abs(cy - sy) > 10:
                move_mouse(sx, sy)
                time.sleep(0.08)
            send_click()
            time.sleep(1.0)
            if not is_overlay_open(page)['open']:
                return True
        except Exception as e:
            log(f"  SendInput 关闭失败: {e}")

    return not is_overlay_open(page)['open']

def extract_data(page, note_id):
    """提取浮层中的笔记数据（纯 DOM 路径 — XHS 浮层不触发 SSR 更新）。"""
    dom_raw = page.run_js("""
        const c = document.querySelector('#noteContainer');
        if (!c) return null;
        const title = c.querySelector('.title, [class*="title"], h1')?.textContent?.trim() || '';
        const desc = c.querySelector('.desc, [class*="desc"], .note-text, #detail-desc')?.textContent?.trim() || '';
        const authorEl = c.querySelector('[class*="author"] [class*="name"], .username, a[href*="/user/"]');
        const author = authorEl?.textContent?.trim() || '';
        // 作者 ID: 从链接 /user/profile/<id> 提取
        let authorId = '';
        const authorLink = c.querySelector('a[href*="/user/profile/"]');
        if (authorLink) {
            const m = authorLink.href.match(/\\/user\\/profile\\/([a-f0-9]+)/);
            if (m) authorId = m[1];
        }
        const likesEl = c.querySelector('[class*="like"] [class*="count"], .like-wrapper .count, [class*="like"] span');
        const collectsEl = c.querySelector('[class*="collect"] [class*="count"], [class*="collect"] span');
        const commentEl = c.querySelector('[class*="comment"] [class*="count"], [class*="chat"] [class*="count"]');
        const shareEl = c.querySelector('[class*="share"] [class*="count"]');
        const imgs = []; c.querySelectorAll('.swiper img, .media-container img, [class*="slider"] img, img[src*="xhscdn"]').forEach(i => {
            const s = i.src || ''; if (s && !s.includes('avatar') && s.includes('xhscdn')) imgs.push(s);
        });
        // 发布时间
        let pubTime = '';
        const dateEl = c.querySelector('.date, [class*="date"], .publish-date, .create-time, [class*="bottom"] span');
        if (dateEl) pubTime = dateEl.textContent.trim();
        if (!pubTime) {
            const m = (c.textContent || '').match(/(\\d{4}[-/]\\d{1,2}[-/]\\d{1,2})/);
            if (m) pubTime = m[1];
        }
        // 话题标签
        const hashtags = [];
        c.querySelectorAll('[class*="tag"], [class*="hashtag"], a[href*="/tag/"], #topic-tag').forEach(t => {
            const name = t.textContent.trim().replace(/^#/, '');
            if (name && !hashtags.includes(name)) hashtags.push(name);
        });
        // 视频链接
        let videoUrl = '';
        const videoEl = c.querySelector('video');
        if (videoEl) {
            videoUrl = videoEl.src || videoEl.querySelector('source')?.src || '';
        }

        return JSON.stringify({
            title, desc, author, authorId,
            likes: likesEl ? parseInt(likesEl.textContent.replace(/[^0-9]/g,'')) || 0 : 0,
            collects: collectsEl ? parseInt(collectsEl.textContent.replace(/[^0-9]/g,'')) || 0 : 0,
            comments: commentEl ? parseInt(commentEl.textContent.replace(/[^0-9]/g,'')) || 0 : 0,
            shares: shareEl ? parseInt(shareEl.textContent.replace(/[^0-9]/g,'')) || 0 : 0,
            images: imgs.slice(0, 18),
            pubTime, hashtags, videoUrl,
        });
    """)
    if dom_raw:
        try:
            d = json.loads(dom_raw)
            if d and d.get('title'):
                return {
                    'note_id': note_id, 'title': d['title'], 'content': d.get('desc',''),
                    'author_name': d.get('author',''), 'author_id': d.get('authorId',''),
                    'liked_count': d.get('likes',0), 'collected_count': d.get('collects',0),
                    'comment_count': d.get('comments',0), 'share_count': d.get('shares',0),
                    'publish_time': d.get('pubTime',''),
                    'images': d.get('images',[]), 'video_url': d.get('videoUrl',''),
                    'hashtags': d.get('hashtags',[]),
                    '_source': 'dom',
                }
        except: pass
    return None


# =============================================================================
# 校准校验工具
# =============================================================================

def _ensure_calibration(calib, page):
    """
    校验校准数据是否仍然有效，窗口移动时自动刷新。

    返回: (calib, changed)
    """
    hwnd = calib.get('hwnd')
    if not hwnd:
        return calib, False
    if not user32.IsWindow(hwnd):
        new_c = calibrate(page)
        return new_c, True

    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    cx, cy, cw, ch = r.left, r.top, r.right - r.left, r.bottom - r.top
    moved = (
        abs(cx - calib.get('wx', cx)) > 5 or
        abs(cy - calib.get('wy', cy)) > 5 or
        abs(cw - calib.get('ww', cw)) > 5 or
        abs(ch - calib.get('wh', ch)) > 5
    )
    if not moved:
        return calib, False

    new_c = calibrate(page)
    return new_c, True


# =============================================================================
# 主流程
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", default="牙痛")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pool", nargs="?", const="auto", default=None, help="archive.db 路径，启用后跳过已采集的帖子")
    args = parser.parse_args()

    # 加载数据池
    pool_ids = set()
    pool_conn = None
    if args.pool:
        if args.pool == "auto":
            args.pool = str(OUT_ROOT / "archive.db")
        pool_path = Path(args.pool)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pool_conn = sqlite3.connect(str(pool_path))
            pool_conn.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, keyword TEXT, note_id TEXT UNIQUE, title TEXT, content TEXT, author_name TEXT, author_id TEXT, liked_count INTEGER DEFAULT 0, collected_count INTEGER DEFAULT 0, comment_count INTEGER DEFAULT 0, share_count INTEGER DEFAULT 0, publish_time TEXT, images TEXT, video_url TEXT, hashtags TEXT, source_url TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            for row in pool_conn.execute("SELECT note_id FROM notes"):
                if row[0]: pool_ids.add(row[0])
            log(f"数据池: {pool_path.name} ({len(pool_ids)} 条已采集)")
        except Exception as e:
            log(f"数据池加载失败: {e}")
            pool_conn = None

    out_dir = Path(args.output) if args.output else OUT_ROOT / f"collect_{args.keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"输出: {out_dir}")

    # 1. 连接 Chrome
    page = connect_chrome()
    search_url = f"https://www.xiaohongshu.com/search_result?keyword={quote(args.keyword)}"
    page.get(search_url, retry=2, timeout=25)
    try: page.wait.ele_displayed('a[href*="/explore/"], a.cover', timeout=20)
    except: pass
    page.wait(2, 4)
    try:
        log(f"页面: {page.title[:60]}")
    except Exception:
        log("页面: (title 获取失败，继续...)")

    # 2. 校准（重试：窗口标题可能延迟出现）
    calib = None
    for _ in range(5):
        calib = calibrate(page)
        if calib.get('hwnd'):
            break
        time.sleep(0.5)
    log(f"校准: hwnd={calib.get('hwnd')} cs=({calib['csx']},{calib['csy']}) dpr={calib['dpr']}")
    if calib.get('hwnd'): force_foreground(calib['hwnd'])
    else: log("  WARN: 未找到 Chrome 窗口句柄，物理点击策略将不可用")

    # 3. 采集循环
    collected = []
    collected_ids = set()
    skipped_ids = set()
    scroll_count = 0
    fail_streak = 0

    while len(collected) < args.count and scroll_count < 2000:
        # 校验校准数据（窗口移动后自动刷新）
        new_calib, calib_changed = _ensure_calibration(calib, page)
        if calib_changed:
            calib = new_calib
            log(f"校准刷新: cs=({calib['csx']},{calib['csy']})")
            if calib.get('hwnd'): force_foreground(calib['hwnd'])

        cards = get_cards(page)
        new_cards = [c for c in cards if c['note_id'] not in collected_ids
                     and c['note_id'] not in pool_ids
                     and c['note_id'] not in skipped_ids]

        if not new_cards:
            scroll_count += 1
            log(f"\n>>> 滚动加载 ({scroll_count}) — 当前无新卡片 <<<")
            try: page.scroll.down(random.randint(500, 800))
            except: pass
            time.sleep(random.uniform(2.0, 4.0))
            continue

        log(f"\n>>> 可见:{len(cards)} 新:{len(new_cards)} 已采:{len(collected)}/{args.count} <<<")

        for card in new_cards:
            if len(collected) >= args.count: break

            nid = card['note_id']
            nth = len(collected) + 1
            log(f"[{nth}/{args.count}] {nid} | {card.get('title','')[:50]}")
            card_failed = False

            # 打开浮层
            if not open_card(page, card):
                log(f"  FAIL: 打开浮层失败")
                skipped_ids.add(nid)
                fail_streak += 1
                card_failed = True

            if not card_failed:
                # 检查浮层
                overlay = is_overlay_open(page)
                if not overlay['open']:
                    log(f"  FAIL: 浮层未打开")
                    skipped_ids.add(nid)
                    fail_streak += 1
                    card_failed = True

            if not card_failed:
                # 提取数据
                note_data = extract_data(page, nid)
                if note_data:
                    collected.append(note_data)
                    collected_ids.add(nid)

                    note_dir = out_dir / f"note_{nth:03d}_{nid}"
                    note_dir.mkdir(parents=True, exist_ok=True)
                    save_json(note_dir / "note_parsed.json", note_data)
                    try: take_screenshot(page, note_dir / "screenshot.png")
                    except: pass

                    # 写入数据池
                    if pool_conn:
                        try:
                            pool_conn.execute(
                                "INSERT OR IGNORE INTO notes(keyword,note_id,title,content,author_name,author_id,liked_count,collected_count,comment_count,share_count,publish_time,images,video_url,hashtags) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (args.keyword, note_data['note_id'], note_data.get('title',''), note_data.get('content',''), note_data.get('author_name',''), note_data.get('author_id',''), note_data['liked_count'], note_data['collected_count'], note_data.get('comment_count',0), note_data.get('share_count',0), note_data.get('publish_time',''), json.dumps(note_data.get('images',[]), ensure_ascii=False), note_data.get('video_url',''), json.dumps(note_data.get('hashtags',[]), ensure_ascii=False)))
                            pool_conn.commit()
                        except Exception as e:
                            log(f"  数据池写入失败: {e}")

                    log(f"  OK: {note_data.get('title','')[:60]}")
                    log(f"  likes={note_data['liked_count']} collects={note_data['collected_count']} imgs={len(note_data.get('images',[]))}")
                    fail_streak = 0
                else:
                    log(f"  FAIL: 数据提取失败")
                    skipped_ids.add(nid)
                    fail_streak += 1

            # 关浮层（无论提取成功或失败都尝试关闭）
            overlay_still_open = is_overlay_open(page)
            if overlay_still_open.get('open'):
                closed = close_overlay(page, calib)
                if not closed:
                    log(f"  WARN: 浮层关闭失败，强制跳过")
                    # 滚动一下改变上下文
                    try: page.scroll.down(random.randint(200, 400))
                    except: pass
                    time.sleep(1)
                    continue

            # 统一 fail_streak 处理：连续失败超过阈值就滚动换区域
            if fail_streak > 3:
                log("  连续失败超过 3 次，滚动换区域...")
                try: page.scroll.down(random.randint(400, 600))
                except: pass
                time.sleep(2)
                fail_streak = 0

            # 人类间隔
            delay = random.uniform(1.5, 3.0)
            log(f"  等待 {delay:.1f}s...")
            time.sleep(delay)

    # 4. 汇总
    log(f"\n{'='*60}")
    log(f"采集完成: {len(collected)}/{args.count} 条")
    log(f"输出: {out_dir}")
    save_json(out_dir / "summary.json", {
        'keyword': args.keyword,
        'total': len(collected),
        'note_ids': [d['note_id'] for d in collected],
        'titles': [d.get('title','')[:60] for d in collected],
    })
    save_json(out_dir / "all_notes.json", collected)
    if pool_conn: pool_conn.close()
    log("浏览器保持运行。Done.")
    return page

if __name__ == "__main__":
    main()
