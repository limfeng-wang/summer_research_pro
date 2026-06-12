"""
collector_engine.py — 增强采集引擎
=====================================
基于 xhs_collector_v2.py 已验证的模式，新增：
  - scroll_to_bottom() — 全滚动到底（连续 N 次无新卡片才停止）
  - PageDisconnectedError — 指数退避重连
  - 数据去重 — SQLite pool_ids 过滤

用法:
    engine = CollectionEngine(config, chrome_manager, store)
    notes = engine.collect_notes("牙痛", count=30, pool_ids=set())
"""

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path


# =============================================================================
# 日志
# =============================================================================

def log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    try:
        print(f"[{ts}] [collector] {msg}", flush=True)
    except Exception:
        pass


# =============================================================================
# 采集引擎
# =============================================================================

# 注意: from xhs_collector_v2 import ... 需要放在运行时导入，因为
# xhs_collector_v2.py 里有 Windows API 的顶层代码。
# 我们在方法内按需 import，避免 import 时崩溃。

_COLLECTOR_MODULE = None


def _get_collector():
    """延迟导入 xhs_collector_v2，避免 Windows API 初始化冲突。"""
    global _COLLECTOR_MODULE
    if _COLLECTOR_MODULE is None:
        # xhs_collector_v2.py 内部用 from xhs_snapshot import ...，
        # 需要 src 目录在 sys.path 中
        _src = str(Path(__file__).resolve().parent)
        if _src not in sys.path:
            sys.path.insert(0, _src)
        import xhs_collector_v2 as m
        _COLLECTOR_MODULE = m
    return _COLLECTOR_MODULE


_SNAPSHOT_MODULE = None


def _get_snapshot():
    """延迟导入 xhs_snapshot。"""
    global _SNAPSHOT_MODULE
    if _SNAPSHOT_MODULE is None:
        try:
            from src import xhs_snapshot as m
        except ImportError:
            import xhs_snapshot as m
        _SNAPSHOT_MODULE = m
    return _SNAPSHOT_MODULE


# =============================================================================
# Window API（仅关闭浮层时用到）
# =============================================================================

_WIN32_MODULE = None


def _get_win32():
    global _WIN32_MODULE
    if _WIN32_MODULE is None:
        try:
            from src import win32_api as m
        except ImportError:
            import win32_api as m
        _WIN32_MODULE = m
    return _WIN32_MODULE


# =============================================================================
# 采集引擎
# =============================================================================

class CollectionEngine:
    """
    增强采集引擎。

    核心流程（从 xhs_collector_v2.py 继承）：
      连接 → 导航 → 坐标校准 → 循环：
        卡片检测 → 过滤已采集 → 无卡片则滚动 → CDP 点击 → 提取 → 保存 → 关闭
    """

    def __init__(self, config, chrome_manager, store):
        self._config = config
        self._chrome_mgr = chrome_manager
        self._store = store
        self._page = None
        self._calib = None
        self._out_dir = None

    # ---- 公共接口 ----

    def collect_notes(self, keyword: str, count: int, pool_ids: set = None) -> list:
        """
        执行一次采集会话。

        参数:
            keyword: 搜索关键词
            count:    本次最多采集数（安全上限）
            pool_ids: 已采集的 ID 集合（用于去重）

        返回:
            [(note_id, title), ...] 采集成功的笔记列表
        """
        self._page = self._chrome_mgr.ensure_connection()

        # Session 过期检查
        expired, reason = self._chrome_mgr.detect_session_expiry(self._page)
        if expired:
            self._chrome_mgr.notify_user(
                f"Session 可能已过期，请检查并重新登录。\n原因: {reason}",
                title="xhs-scraper: Session 过期"
            )
            return []

        # 检查上次运行是否留下了 session_expired.flag（仅在本次会话有效时提醒）
        flag = self._chrome_mgr.read_expiry_flag()
        if flag is not None:
            log(f"检测到之前的 session 过期标记，建议重新登录")
            self._chrome_mgr.clear_expiry_flag()

        # 准备输出目录
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_root = Path(self._config.collection.data_dir) / f"collect_{keyword}_{ts}"
        out_root.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_root

        # 导航到搜索页
        from urllib.parse import quote
        search_url = (
            f"https://www.xiaohongshu.com/search_result"
            f"?keyword={quote(keyword)}"
        )
        self._search_url = search_url
        log(f"导航到搜索页: {search_url}")
        try:
            self._page.get(search_url, retry=2, timeout=25)
            self._human_wait(2, 4)
        except Exception as e:
            log(f"导航失败: {e}")
            self._page = self._chrome_mgr.reconnect(self._page)
            return []

        # 等待页面加载
        try:
            self._page.wait.ele_displayed('a[href*="/explore/"]', timeout=15)
        except Exception:
            log("警告: 搜索页卡片加载可能超时")

        # 坐标校准
        win32 = _get_win32()
        self._calib = win32.calibrate(self._page)
        log(f"校准完成: DPR={self._calib['dpr']:.1f}")

        collected = []
        collected_ids = set(pool_ids or [])
        fail_streak = 0
        scroll_count = 0
        max_scrolls = self._config.session_limits.max_scrolls
        stale_limit = self._config.session_limits.scroll_stale_limit

        col_mod = _get_collector()

        while len(collected) < count and scroll_count < max_scrolls:
            # 检测 Session 过期（每 10 条检查一次）
            if len(collected) > 0 and len(collected) % 10 == 0:
                expired, _ = self._chrome_mgr.detect_session_expiry(self._page)
                if expired:
                    log("Session 已过期，停止采集")
                    break

            # 1. 获取可见卡片
            try:
                cards = col_mod.get_cards(self._page)
            except Exception as e:
                log(f"get_cards 失败: {e}")
                if self._handle_disconnect(e):
                    continue
                break

            # 2. 过滤已采集
            new_cards = [
                c for c in cards
                if c['note_id'] not in collected_ids
            ]

            if not new_cards:
                # 3. 无新卡片 → 滚动
                self._scroll_to_bottom(keyword, col_mod)

                # 滚动后重新检查是否有未采集的卡片
                try:
                    cards = col_mod.get_cards(self._page)
                    new_cards = [
                        c for c in cards
                        if c['note_id'] not in collected_ids
                    ]
                except Exception:
                    new_cards = []

                if not new_cards:
                    scroll_count += 1
                    if scroll_count >= stale_limit:
                        log(f"连续 {stale_limit} 次滚动无新卡片，判定为到底")
                        break
                    self._human_wait(2, 4)
                    continue
                else:
                    scroll_count = 0
                    # 有新卡片 → 进入下面的处理流程

            # 4. 处理新卡片
            scroll_count = 0
            fail_streak = 0

            for card in new_cards:
                if len(collected) >= count:
                    break

                nid = card['note_id']
                log(f"\n[{len(collected) + 1}/{count}] note_id: {nid}")

                try:
                    # CDP 点击打开浮层
                    opened = col_mod.open_card(self._page, card)
                    if not opened:
                        log("  打开浮层失败")
                        continue

                    # 确认浮层已打开
                    overlay = col_mod.is_overlay_open(self._page)
                    if not overlay.get('open'):
                        log("  浮层未打开")
                        continue

                    # 提取数据（SSR 优先，DOM 降级）
                    note_data = col_mod.extract_data(self._page, nid)
                    if not note_data:
                        log("  提取数据失败")
                        col_mod.close_overlay(self._page, self._calib)
                        self._human_wait(1, 2)
                        continue

                    # 保存
                    self._save_note(nid, note_data, len(collected) + 1)

                    # 写 SQLite 去重
                    try:
                        self._store.mark_note_collected(keyword, nid)
                    except Exception:
                        pass

                    collected_ids.add(nid)
                    collected.append((nid, note_data.get('title', '')))
                    fail_streak = 0

                    # 关闭浮层
                    try:
                        col_mod.close_overlay(self._page, self._calib)
                    except Exception as e:
                        log(f"关闭浮层失败: {e}")

                except Exception as e:
                    log(f"处理卡片异常: {e}")
                    if self._handle_disconnect(e):
                        # 重连后重新导航到搜索页
                        try:
                            self._page.get(search_url, retry=2, timeout=25)
                            self._human_wait(2, 3)
                        except Exception:
                            pass
                        break
                    fail_streak += 1
                    if fail_streak > 3:
                        log("连续失败超过 3 次，滚动换区域")
                        try:
                            self._page.scroll.down(random.randint(400, 600))
                            time.sleep(random.uniform(2.0, 3.0))
                        except Exception:
                            pass
                        fail_streak = 0
                    continue

                # 人类行为间隔
                self._human_wait(1.5, 3.5)

            # 每批卡片完成后休息
            if len(collected) > 0 and len(collected) % 5 == 0 and len(collected) < count:
                rest = random.uniform(8, 20)
                log(f"采集 {len(collected)} 条，休息 {rest:.0f}s...")
                time.sleep(rest)

        # 保存汇总
        self._save_summary(keyword, collected)
        log(f"\n{'='*40}")
        log(f"采集完成: {len(collected)} 条 (关键词: {keyword})")

        # 清理多余标签页，释放内存
        try:
            self._chrome_mgr.cleanup_tabs(self._page)
        except Exception:
            pass

        return collected

    # ---- 滚动逻辑 ----

    def _scroll_to_bottom(self, keyword: str, col_mod) -> int:
        """
        全滚动：持续滚动直到连续 N 次无新卡片。

        返回本次滚动新增的卡片数量。
        """
        stale_limit = self._config.session_limits.scroll_stale_limit
        max_scrolls = self._config.session_limits.max_scrolls

        previous_ids = self._get_visible_ids(col_mod)
        stale = 0

        for attempt in range(max_scrolls):
            amount = random.randint(300, 800)
            try:
                self._page.scroll.down(amount)
            except Exception as e:
                if self._handle_disconnect(e):
                    return 0
            self._human_wait(2.0, 4.0)

            current_ids = self._get_visible_ids(col_mod)
            new_ids = current_ids - previous_ids

            if not new_ids:
                stale += 1
                log(f"  滚动 {attempt + 1}/{max_scrolls}: 无新卡片 (停滞 {stale}/{stale_limit})")
                if stale >= stale_limit:
                    log(f"  判定到底: 连续 {stale_limit} 次无新卡片")
                    break
            else:
                stale = 0
                previous_ids = current_ids
                log(f"  滚动 {attempt + 1}: 发现 {len(new_ids)} 张新卡片")

        return len(previous_ids)

    def _get_visible_ids(self, col_mod) -> set:
        """获取当前可见卡片的 ID 集合。"""
        try:
            cards = col_mod.get_cards(self._page)
            return {c['note_id'] for c in cards if c.get('note_id')}
        except Exception:
            return set()

    # ---- 断线重连 ----

    def _handle_disconnect(self, error: Exception, search_url: str = None) -> bool:
        """
        判断错误是否为连接断开，若是则自动重连并恢复状态。

        返回 True 表示已重连成功，调用方可继续。
        """
        err_name = type(error).__name__
        err_msg = str(error)
        haystack = err_name + " " + err_msg
        if any(k in haystack for k in ("Disconnect", "Timeout", "Broken", "Connection")):
            log(f"检测到连接问题: {err_name}")
            try:
                self._page = self._chrome_mgr.reconnect(self._page)

                # 恢复导航页面
                ctx = getattr(self._chrome_mgr, '_reconnect_context', {})
                target_url = search_url or ctx.get('url') or getattr(self, '_search_url', None)
                if target_url:
                    try:
                        self._page.get(target_url, retry=2, timeout=25)
                        self._human_wait(2, 3)
                    except Exception as e:
                        log(f"重连后导航恢复失败: {e}")

                # 恢复滚动位置
                scroll_y = ctx.get('scroll_y', 0)
                if scroll_y:
                    try:
                        self._page.scroll.to(scroll_y)
                    except Exception:
                        pass

                # 重新校准视口坐标
                win32 = _get_win32()
                self._calib = win32.calibrate(self._page)
                log("重连+状态恢复完成")
                return True
            except Exception as re:
                log(f"重连失败: {re}")
        return False

    # ---- 保存 ----

    def _save_note(self, note_id: str, data: dict, nth: int):
        """保存笔记数据到文件。"""
        note_dir = self._out_dir / f"note_{nth:03d}_{note_id}"
        note_dir.mkdir(parents=True, exist_ok=True)

        snap = _get_snapshot()
        snap.save_json(note_dir / "note_parsed.json", data)

        try:
            snap.take_screenshot(self._page, note_dir / "screenshot.png")
        except Exception as e:
            log(f"  截图失败: {e}")

        title = data.get('title', '')[:60]
        log(f"  ✓ 已保存: {title}")

    def _save_summary(self, keyword: str, collected: list):
        """保存会话汇总。"""
        summary = {
            "keyword": keyword,
            "collected": len(collected),
            "time": datetime.now().isoformat(),
            "notes": [{"id": nid, "title": title} for nid, title in collected],
        }
        snap = _get_snapshot()
        snap.save_json(self._out_dir / "summary.json", summary)

    # ---- 人类行为模拟 ----

    def _human_wait(self, min_s: float, max_s: float):
        """等待随机时长，模拟人类。"""
        delay = random.uniform(min_s, max_s)
        time.sleep(delay)
