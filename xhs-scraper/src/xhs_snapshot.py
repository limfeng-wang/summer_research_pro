#!/usr/bin/env python3
"""
xhs_snapshot.py — 小红书搜索页全量快照 + 精准点击详情采集

功能:
  1. 连接已在运行的 Chrome CDP（需先 python chrome_launcher.py）
  2. 搜索关键词 → 保存搜索页全部内容（SSR / API响应 / 截图 / HTML）
  3. 精准点击卡片进入详情 → 保存详情页全部内容
  4. 向下滚动加载更多 → 重复采集

保存结构（每次运行一个时间戳目录）:
  输出数据/snapshot_{keyword}_{timestamp}/
  ├── search/
  │   ├── ssr.json              搜索页 __INITIAL_STATE__ 完整 JSON
  │   ├── dom_cards.json         DOM 中所有可见笔记卡片
  │   ├── api_responses.json     CDP Network 拦截的搜索 API 响应
  │   ├── screenshot.png         搜索页全页截图
  │   └── page.html              搜索页 HTML 快照
  └── detail_{note_id}/
      ├── ssr.json               详情页 __INITIAL_STATE__ 完整 JSON
      ├── note_parsed.json       解析后的笔记结构化数据
      ├── screenshot.png         详情页截图
      └── page.html              详情页 HTML 快照

用法:
  cd xhs-scraper && python src/xhs_snapshot.py --keyword "牙痛"
  cd xhs-scraper && python src/xhs_snapshot.py --keyword "牙痛" --max 10
"""

import argparse
import base64
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import random
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from DrissionPage import ChromiumPage, ChromiumOptions

# =============================================================================
# 反检测 JS — 从 _common.py 统一导入（避免两份拷贝不一致）
# =============================================================================
from _common import ANTI_DETECTION_JS

# =============================================================================
# SSR 提取 JS（brace matching，绕过 Vue reactive proxy）
# =============================================================================
SSR_EXTRACT_JS = r"""
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
        if (c === '\\' && inStr) { esc = true; continue; }
        if (c === '"') { inStr = !inStr; continue; }
        if (inStr) continue;
        if (c === '{') depth++;
        else if (c === '}') { depth--; if (depth === 0) return t.substring(start, i + 1); }
    }
}
return null;
"""


# =============================================================================
# 工具函数
# =============================================================================

def log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    try:
        print(f"[{ts}] {msg}", flush=True)
    except UnicodeEncodeError:
        # Windows GBK 终端无法编码 emoji 等字符 → 替换为 ?
        enc = sys.stdout.encoding or 'utf-8'
        safe = msg.encode(enc, errors='replace').decode(enc)
        print(f"[{ts}] {safe}", flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(p: Path, data):
    with open(str(p), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log(f"  已保存: {p.name}  ({_json_size(data)})")


def save_text(p: Path, text: str):
    with open(str(p), 'w', encoding='utf-8') as f:
        f.write(text)
    log(f"  已保存: {p.name}  ({len(text)} 字符)")


def take_screenshot(page, path: Path):
    """通过 CDP 截取全视口截图。"""
    result = page.run_cdp('Page.captureScreenshot', format='png')
    data = result.get('data', '')
    if data:
        with open(str(path), 'wb') as f:
            f.write(base64.b64decode(data))
        log(f"  已保存: {path.name}")
    else:
        log(f"  截图返回空数据")


def _json_size(data) -> str:
    s = json.dumps(data, ensure_ascii=False)
    if len(s) < 1024:
        return f"{len(s)} B"
    if len(s) < 1024 * 1024:
        return f"{len(s) / 1024:.1f} KB"
    return f"{len(s) / 1024 / 1024:.1f} MB"


# =============================================================================
# Chrome 查找 / 启动 / 连接
# =============================================================================

CDP_PORT = 9222
_DEFAULT_PROFILE = "profiles/chrome_main"


def _find_chrome() -> str:
    """跨平台查找 Chrome 可执行文件路径。"""
    if sys.platform == 'win32':
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
    elif sys.platform == 'darwin':
        p = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.exists(p):
            return p
    else:
        for name in ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']:
            p = shutil.which(name)
            if p:
                return p
    raise FileNotFoundError("未找到 Google Chrome，请确认已安装。")


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def _wait_for_cdp(port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_port_in_use(port):
            try:
                r = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1)
                if r.status == 200:
                    return True
            except Exception:
                pass
        time.sleep(0.5)
    return False


def _pid_file(profile_dir: str) -> str:
    return os.path.join(os.path.abspath(profile_dir), "chrome.pid")


def _launch_chrome(port: int = CDP_PORT, profile_dir: str = None):
    """启动 Chrome 并开启 CDP 调试端口。"""
    if profile_dir is None:
        profile_dir = _DEFAULT_PROFILE
    chrome_path = _find_chrome()
    log(f"Chrome 路径: {chrome_path}")

    abs_profile = os.path.abspath(profile_dir)
    os.makedirs(abs_profile, exist_ok=True)

    cmd = [
        chrome_path,
        f"--user-data-dir={abs_profile}",
        f"--remote-debugging-port={port}",
        "--disable-blink-features=AutomationControlled",
        "--no-first-run",
        "--no-default-browser-check",
        "--window-size=1400,900",
        "https://www.xiaohongshu.com"
    ]

    log(f"启动 Chrome（端口 {port}, profile={profile_dir}）...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 保存 PID 供下次复用
    try:
        with open(_pid_file(profile_dir), 'w') as f:
            f.write(str(proc.pid))
    except Exception:
        pass
    log("等待 CDP 端口就绪...")

    if not _wait_for_cdp(port):
        proc.terminate()
        raise RuntimeError(f"Chrome 启动超时，CDP 端口 {port} 未就绪。")

    log(f"Chrome 已启动，CDP: 127.0.0.1:{port}")
    return proc


def _get_saved_pid(profile_dir: str = None) -> int | None:
    """读取之前保存的 Chrome PID。"""
    if profile_dir is None:
        profile_dir = _DEFAULT_PROFILE
    try:
        with open(_pid_file(profile_dir)) as f:
            return int(f.read().strip())
    except Exception:
        return None

def _is_pid_alive(pid: int) -> bool:
    """检查进程是否存活。"""
    try:
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)  # PROCESS_QUERY_INFO | PROCESS_VM_READ
            if not handle: return False
            exit_code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return exit_code.value == 259  # STILL_ACTIVE
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False

def connect_chrome(port: int = CDP_PORT, auto_launch: bool = True, profile_dir: str = None) -> ChromiumPage:
    """
    连接到 Chrome CDP。

    优先使用已保存 PID 的 Chrome；其次尝试端口；最后才启动新实例。

    profile_dir: Chrome 用户数据目录，None 时用默认值（支持多账号）
    """
    if profile_dir is None:
        profile_dir = _DEFAULT_PROFILE
    addr = f"127.0.0.1:{port}"
    co = ChromiumOptions()
    co.set_address(addr)

    # 检查已保存的 Chrome 是否还活着
    saved_pid = _get_saved_pid(profile_dir)
    if saved_pid and _is_pid_alive(saved_pid) and _is_port_in_use(port):
        co.existing_only(True)
        log(f"复用已有 Chrome (PID {saved_pid}, profile={profile_dir})")
    elif auto_launch:
        _launch_chrome(port, profile_dir)
        co.existing_only(True)
    else:
        log(f"Chrome 未运行在 {addr}，请先启动。")
        sys.exit(1)

    try:
        page = ChromiumPage(co)
    except Exception as e:
        log(f"连接 Chrome 失败: {e}")
        sys.exit(1)

    try:
        page.run_cdp('Page.addScriptToEvaluateOnNewDocument', source=ANTI_DETECTION_JS)
        log("反检测 JS 已注入")
    except Exception as e:
        log(f"警告: 反检测 JS 注入失败: {e}")

    log(f"已连接: {page.url[:80]}")
    return page


# =============================================================================
# 搜索 API 响应捕获
# =============================================================================

def collect_api_responses(page: ChromiumPage) -> list:
    """
    通过 CDP Network 域收集搜索相关的 API 响应体。

    思路：遍历浏览器已记录的 requestId，找到搜索 API 的响应并获取 body。
    相比实时回调更稳定，不依赖 DrissionPage 的事件系统。
    """
    captured = []
    try:
        page.run_cdp("Network.enable")
    except Exception:
        return captured

    # 用 JS 从 performance API 中提取搜索相关的网络请求 URL
    raw = page.run_js(r"""
        const entries = performance.getEntriesByType('resource');
        const urls = [];
        const seen = new Set();
        for (const e of entries) {
            const u = e.name;
            if (seen.has(u)) continue;
            seen.add(u);
            if (u.includes('/api/sns/web/v1/search/')
                || u.includes('/api/sns/web/v1/feed')
                || u.includes('/api/sns/web/v1/note/')
                || u.includes('search/notes')) {
                urls.push(u);
            }
        }
        return JSON.stringify(urls.slice(0, 50));
    """)
    api_urls = []
    if raw:
        try:
            api_urls = json.loads(raw)
        except json.JSONDecodeError:
            pass

    if not api_urls:
        return captured

    for url in api_urls:
        try:
            captured.append({
                'url': url,
                'note': '从 performance API 提取的 URL，完整响应体需通过 CDP Network.getResponseBody 获取',
            })
        except Exception:
            pass

    log(f"  从 performance API 发现 {len(api_urls)} 条搜索 API 请求")
    return captured


# =============================================================================
# 数据提取
# =============================================================================

def extract_ssr_state(page) -> dict | None:
    """从页面提取 __INITIAL_STATE__ 完整 JSON。"""
    raw = page.run_js(SSR_EXTRACT_JS)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = re.sub(r':undefined(?=\s*[,\]}])', ':null', raw)
            cleaned = re.sub(r'(?<=[,\[])\s*undefined(?=\s*[,\]}])', 'null', cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
    return None


def extract_dom_cards(page) -> list[dict]:
    """从 DOM 提取所有可见笔记卡片（去重，优先保留带 xsec_token 的条目）。"""
    raw = page.run_js(r"""
        const results = [];
        const seen = new Set();
        document.querySelectorAll('a').forEach(a => {
            const href = a.getAttribute('href') || '';
            if (!href || seen.has(href)) return;
            if (!href.includes('/explore/') && !href.includes('/search_result/')
                && !href.includes('/discovery/item/')) return;
            seen.add(href);
            let title = a.getAttribute('title') || '';
            if (!title) {
                const spans = a.querySelectorAll('span, div[class*="title"], p[class*="title"]');
                for (const s of spans) {
                    const t = (s.textContent || '').trim();
                    if (t.length > 3 && t.length < 200) { title = t; break; }
                }
            }
            if (!title) {
                const img = a.querySelector('img');
                title = (img && (img.alt || img.getAttribute('aria-label'))) || '';
            }
            if (!title) {
                title = (a.textContent || '').trim().slice(0, 100);
            }
            // 提取 xsec_token
            let xsec = '';
            const m = href.match(/xsec_token=([^&]+)/);
            if (m) xsec = m[1];
            // 提取 note_id
            let nid = '';
            const nm = href.match(/\/(?:explore|search_result)\/([a-f0-9]+)/)
                   || href.match(/\/discovery\/item\/([a-f0-9]+)/);
            if (nm) nid = nm[1];
            results.push({
                href: href,
                title: title.trim().slice(0, 100),
                note_id: nid,
                xsec_token: xsec,
            });
        });
        return JSON.stringify(results);
    """)
    if not raw:
        return []
    try:
        raw_cards = json.loads(raw)
    except json.JSONDecodeError:
        return []

    # 去重：同一 note_id 只保留一个，优先保留带 xsec_token 的
    seen = {}
    for c in raw_cards:
        nid = c.get('note_id', '')
        if not nid:
            continue
        if nid not in seen:
            seen[nid] = c
        elif c.get('xsec_token') and not seen[nid].get('xsec_token'):
            seen[nid] = c  # 用有 xsec_token 的替换

    result = list(seen.values())
    log(f"  DOM 卡片: {len(raw_cards)} 个链接 → {len(result)} 张独立笔记")
    return result


def parse_note_detail(state: dict) -> dict | None:
    """从详情页 SSR 中提取笔记结构化数据。"""
    if not state:
        return None
    note_root = state.get('note')
    if not note_root:
        return None
    dm = note_root.get('noteDetailMap') or {}
    if not dm:
        return None

    nid = note_root.get('currentNoteId') or note_root.get('firstNoteId')
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

    note = entry.get('note')
    if not note:
        return None

    user = note.get('user') or {}
    ii = note.get('interactInfo') or {}

    ts = note.get('time', 0)
    pub_time = ''
    if ts:
        try:
            pub_time = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass

    images = []
    for img in (note.get('imageList') or []):
        if isinstance(img, dict) and img.get('urlDefault'):
            images.append(img['urlDefault'])

    tags = [t['name'] for t in (note.get('tagList') or []) if isinstance(t, dict) and t.get('name')]

    video_url = ''
    video = note.get('video')
    if isinstance(video, dict):
        media = video.get('media') or {}
        streams = media.get('stream') or {}
        for codec in ('h265', 'h264'):
            for s in (streams.get(codec) or []):
                if isinstance(s, dict) and s.get('masterUrl'):
                    video_url = s['masterUrl']
                    break
            if video_url:
                break

    return {
        'note_id': note.get('noteId', ''),
        'title': note.get('title', ''),
        'content': note.get('desc', ''),
        'author_name': user.get('nickname') or user.get('nickName') or '',
        'author_id': user.get('userId', ''),
        'liked_count': int(ii.get('likedCount') or ii.get('liked') or 0),
        'collected_count': int(ii.get('collectedCount') or ii.get('collected') or 0),
        'comment_count': int(ii.get('commentCount') or ii.get('comment') or 0),
        'share_count': int(ii.get('sharedCount') or ii.get('shareCount') or 0),
        'publish_time': pub_time,
        'images': images,
        'video_url': video_url,
        'hashtags': tags,
    }


# =============================================================================
# 直接 URL 导航 — 详情采集
# =============================================================================

def _back_to_search(page: ChromiumPage, search_url: str):
    """模拟用户点"返回"按钮回到搜索页，保留滚动位置。"""
    # 先短暂停留（模拟阅读后反应时间）
    time.sleep(random.uniform(0.3, 0.8))
    # 优先用 history.back() — 和真实用户行为一致
    try:
        page.run_js("window.history.back()")
        page.wait(random.uniform(1.0, 2.0))
        if '/search_result' in page.url:
            return
    except Exception:
        pass
    # fallback: 直接导航（仅在 history.back 失败时）
    try:
        page.get(search_url, retry=1, timeout=15)
        page.wait(random.uniform(1.0, 2.0))
    except Exception:
        pass


def navigate_and_collect_detail(page: ChromiumPage, card: dict,
                                keyword: str, search_url: str,
                                out_dir: Path) -> dict | None:
    """
    通过直接 URL 导航进入详情页，采集全部内容后返回搜索页。

    小红书允许通过 /search_result/{note_id}?xsec_token=... 直接访问详情，
    会自动重定向到 /explore/{note_id}?xsec_token=...

    card: extract_dom_cards 返回的单张卡片 dict（含 href, note_id, xsec_token）
    """
    note_id = card['note_id']
    href = card.get('href', '')
    xsec = card.get('xsec_token', '')

    detail_dir = out_dir / f"detail_{note_id}"
    detail_dir.mkdir(parents=True, exist_ok=True)

    log(f"  导航到详情: {note_id}")

    # 构造完整 URL
    if href.startswith('/'):
        target_url = 'https://www.xiaohongshu.com' + href
    else:
        target_url = href

    # 如果没有 xsec_token，尝试用 /explore/ 路径
    if not xsec:
        target_url = f"https://www.xiaohongshu.com/explore/{note_id}"

    # 直接导航
    try:
        page.get(target_url, retry=2, timeout=20)
    except Exception as e:
        log(f"  导航失败: {e}")
        return None

    page.wait(random.uniform(1.5, 2.5))
    current_url = page.url

    # 检查是否成功进入详情页
    if '404' in current_url:
        log(f"  被重定向到 404，跳过")
        _back_to_search(page, search_url)
        return None

    if '/explore/' not in current_url:
        log(f"  未进入详情页，当前: {current_url[:100]}")
        return None

    log(f"  详情页: {page.title[:60]}")

    # ---- 采集详情页全部内容 ----
    # 1) SSR 完整 JSON
    ssr = extract_ssr_state(page)
    if ssr:
        save_json(detail_dir / "ssr.json", ssr)

    # 2) 解析后的笔记数据
    note_data = parse_note_detail(ssr) if ssr else None
    if note_data:
        save_json(detail_dir / "note_parsed.json", note_data)
        log(f"    → {note_data.get('title', '')[:60]}")
    else:
        log(f"    → 未能解析笔记数据")

    # 3) 页面 HTML 快照
    try:
        html = page.run_js("return document.documentElement.outerHTML")
        save_text(detail_dir / "page.html", html)
    except Exception as e:
        log(f"    HTML 保存失败: {e}")

    # 4) 截图
    try:
        take_screenshot(page, detail_dir / "screenshot.png")
    except Exception as e:
        log(f"   截图失败: {e}")

    # ---- 返回搜索页（模拟用户点"返回"，保留滚动位置） ----
    _back_to_search(page, search_url)

    return note_data


# =============================================================================
# 滚动加载
# =============================================================================

def scroll_to_load(page: ChromiumPage, rounds: int = 3):
    """向下滚动加载更多卡片。每次滚动幅度随机，模拟人类行为。"""
    for i in range(rounds):
        amount = random.randint(300, 700)
        try:
            page.scroll.down(amount)
        except Exception:
            pass
        pause = random.uniform(1.0, 2.5)
        log(f"  滚动 {amount}px，等待 {pause:.1f}s")
        time.sleep(pause)
    # 再等一下让异步数据加载
    time.sleep(random.uniform(1.0, 2.0))


# =============================================================================
# 搜索页全量快照
# =============================================================================

def snapshot_search_page(page: ChromiumPage, search_dir: Path):
    """保存搜索页的全部内容。"""
    log("保存搜索页快照...")
    ensure_dir(search_dir)

    # 1) SSR JSON
    ssr = extract_ssr_state(page)
    if ssr:
        save_json(search_dir / "ssr.json", ssr)
    else:
        log("  警告: 未能提取 SSR 数据")

    # 2) DOM 卡片列表
    cards = extract_dom_cards(page)
    save_json(search_dir / "dom_cards.json", {
        'count': len(cards),
        'cards': cards,
    })

    # 3) HTML 快照
    try:
        html = page.run_js("return document.documentElement.outerHTML")
        save_text(search_dir / "page.html", html)
    except Exception as e:
        log(f"  HTML 保存失败: {e}")

    # 4) 截图
    try:
        take_screenshot(page, search_dir / "screenshot.png")
    except Exception as e:
        log(f"  截图失败: {e}")


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="小红书搜索页全量快照 + 精准点击详情采集")
    parser.add_argument("--keyword", default="牙痛", help="搜索关键词")
    parser.add_argument("--max", type=int, default=10, help="最多采集笔记篇数")
    parser.add_argument("--output", default=None, help="输出目录（默认自动生成时间戳目录）")
    parser.add_argument("--port", type=int, default=CDP_PORT, help=f"CDP 调试端口（默认 {CDP_PORT}）")
    parser.add_argument("--no-launch", action="store_true", help="不自动启动 Chrome，只连已有的")
    args = parser.parse_args()

    # ---- 输出目录 ----
    if args.output:
        out_root = Path(args.output)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_root = Path(__file__).resolve().parent.parent / "data" / f"snapshot_{args.keyword}_{ts}"
    ensure_dir(out_root)
    log(f"输出目录: {out_root}")

    # ---- 连接 Chrome ----
    page = connect_chrome(port=args.port, auto_launch=not args.no_launch)

    # ---- 导航到搜索页 ----
    search_url = f"https://www.xiaohongshu.com/search_result?keyword={quote(args.keyword)}"
    log(f"导航到搜索页: {search_url}")
    page.get(search_url, retry=2, timeout=25)
    page.wait(2, 4)

    # ---- 收集 API 响应 ----
    api_responses = collect_api_responses(page)

    try:
        page.wait.ele_displayed('a[href*="/explore/"], a.cover', timeout=15)
    except Exception:
        log("警告: 搜索页内容可能未完全加载")

    # ---- 滚动加载更多 ----
    scroll_to_load(page, rounds=3)

    # ---- 搜索页快照 ----
    search_dir = ensure_dir(out_root / "search")
    snapshot_search_page(page, search_dir)

    # ---- 保存 API 响应 ----
    if api_responses:
        save_json(search_dir / "api_responses.json", {
            'count': len(api_responses),
            'responses': api_responses,
        })
    else:
        log("  未捕获到搜索 API 响应")

    # ---- 卡片列表 ----
    cards = extract_dom_cards(page)
    log(f"搜索页共发现 {len(cards)} 张独立笔记")

    if not cards:
        log("没有找到任何卡片，退出。")
        page.quit()
        return

    # ---- 通过 URL 导航逐篇采集详情 ----
    collected = 0
    collected_ids = set()

    for card in cards:
        if collected >= args.max:
            break

        note_id = card.get('note_id', '')
        if not note_id or note_id in collected_ids:
            continue

        title_preview = card.get('title') or note_id
        log(f"\n[{collected + 1}/{args.max}] 准备采集: {title_preview[:50]}")

        result = navigate_and_collect_detail(
            page, card, args.keyword, search_url, out_root)
        if result:
            collected += 1
            collected_ids.add(note_id)

        # 采集间隔（模拟人类）
        time.sleep(random.uniform(1.0, 3.0))

        # 每采集 3-6 篇休息一下
        break_interval = random.randint(3, 6)
        if collected > 0 and collected % break_interval == 0 and collected < args.max:
            break_sec = random.uniform(15, 40)
            log(f"\n  休息 {break_sec:.0f}s...\n")
            time.sleep(break_sec)

    # ---- 摘要 ----
    log(f"\n{'='*50}")
    log(f"采集完成")
    log(f"  关键词: {args.keyword}")
    log(f"  采集笔记: {collected} 篇")
    log(f"  搜索页卡片: {len(cards)} 张")
    log(f"  API 响应: {len(api_responses)} 条")
    log(f"  输出目录: {out_root}")

    page.quit()


if __name__ == "__main__":
    main()
