"""
chrome_manager.py — Chrome 生命周期管理器
========================================
职责:
  - Chrome 启动 / 连接 / 复用（委托给 xhs_snapshot.connect_chrome）
  - 连接断开后自动重连（指数退避，最多 3 次）
  - Session 过期检测（登录跳转、滑块验证码、静默空数据、封禁页）
  - 过期时通知用户（Windows 弹窗 + 提示音）

用法:
    mgr = ChromeManager(config)
    page = mgr.ensure_connection()
    if mgr.detect_session_expiry(page)[0]:
        mgr.notify_user("Session 已过期，请重新扫码登录")
"""

import os
import sys
import time
import json
import random
import threading
from datetime import datetime

from src._common import ANTI_DETECTION_JS, GracefulExiter


# =============================================================================
# Session 过期探测器
# =============================================================================

class SessionDetector:
    """
    多策略 Session 过期检测。

    检测策略：
      1. 登录跳转  — 导航到首页，检查 URL 是否包含 /login /cas 等
      2. 滑块验证码 — 检查页面是否存在 #captcha、.captcha、滑块 iframe
      3. 静默空数据 — 检查搜索结果中笔记数量是否为 0
      4. 封禁页     — 检查页面标题/内容是否包含风控提示
    """

    LOGIN_PATTERNS = [
        "/login", "/cas/login", "/passport", "/oauth",
        "login.xiaohongshu.com",
    ]
    CAPTCHA_SELECTORS = [
        "#captcha", ".captcha", "#nc-container",
        ".geetest", "#slideCaptcha", "[class*=\"captcha\"]",
        "iframe[src*=\"captcha\"]", "iframe[src*=\"slider\"]",
    ]
    BLOCK_PAGE_KEYWORDS = [
        "访问受限", "请求过于频繁", "操作太频繁",
        "安全验证", "暂时无法访问", "系统检测到异常",
        "access denied", "rate limit", "too many requests",
    ]

    def __init__(self, page):
        self._page = page

    def check_all(self) -> tuple:
        """
        执行全量检测，返回 (is_expired: bool, reason: str)。
        reason 为空字符串表示未过期。
        """
        # 1. 登录跳转检测
        expired, reason = self._check_login_redirect()
        if expired:
            return True, reason

        # 2. 滑块/验证码检测
        expired, reason = self._check_captcha()
        if expired:
            return True, reason

        # 3. 封禁页检测
        expired, reason = self._check_block_page()
        if expired:
            return True, reason

        # 4. 空数据检测（需要搜索结果页上下文）
        expired, reason = self._check_empty_results()
        if expired:
            return True, reason

        return False, ""

    def _check_login_redirect(self) -> tuple:
        """检测是否被重定向到登录页。"""
        try:
            url = self._page.url
        except Exception:
            return False, ""
        for pattern in self.LOGIN_PATTERNS:
            if pattern in url:
                return True, f"检测到登录跳转: {url[:80]}"
        return False, ""

    def _check_captcha(self) -> tuple:
        """检测页面是否存在滑块/验证码。"""
        for sel in self.CAPTCHA_SELECTORS:
            try:
                el = self._page.run_js(
                    f"document.querySelector('{sel}') !== null"
                )
                if el:
                    return True, f"检测到滑块验证码: {sel}"
            except Exception:
                pass
        return False, ""

    def _check_block_page(self) -> tuple:
        """检测是否被风控拦截（封禁页）。"""
        try:
            title = self._page.title or ""
            body_text = self._page.run_js(
                "document.body?.innerText?.slice(0, 500) || ''"
            ) or ""
            combined = (title + " " + body_text).lower()
            for kw in self.BLOCK_PAGE_KEYWORDS:
                if kw.lower() in combined:
                    return True, f"检测到风控页面: '{kw}'"
        except Exception:
            pass
        return False, ""

    def _check_empty_results(self) -> tuple:
        """
        检测搜索结果是否为空（静默空数据）。
        仅在搜索页时有效。
        """
        try:
            url = self._page.url
            if "/search_result" not in url:
                return False, ""
            # 检测搜索结果容器是否存在且有子元素
            has_results = self._page.run_js(
                "document.querySelectorAll('a[href*=\"/explore/\"]').length > 0"
            )
            if has_results is False:
                return True, "搜索结果页无任何笔记链接（可能已静默封禁）"
        except Exception:
            pass
        return False, ""


# =============================================================================
# Chrome 管理器
# =============================================================================

class ChromeManager:
    """
    Chrome 生命周期管理器。

    不直接启动 Chrome 进程，而是委托给 xhs_snapshot.connect_chrome()。
    不修改 xhs_snapshot.py 中的任何代码。
    """

    def __init__(self, config, log_fn=None):
        self.config = config
        self._page = None
        self._log = log_fn or _default_log
        self._exiter = GracefulExiter()
        # WebSocket keepalive — 每 15s 发送 Browser.getVersion 防止空闲断开
        self._keepalive_interval = 15
        self._keepalive_thread = None
        self._keepalive_stop = threading.Event()

    # ---- WebSocket keepalive ----

    def _start_keepalive(self):
        """启动守护线程，定期发送 CDP 轻量命令保持 WebSocket 活跃。"""
        self._stop_keepalive()
        self._keepalive_stop.clear()

        def _ping():
            while not self._keepalive_stop.is_set():
                try:
                    if self._page is not None:
                        self._page.run_cdp("Browser.getVersion")
                except Exception:
                    break  # 连接已断开，退出线程
                self._keepalive_stop.wait(self._keepalive_interval)

        t = threading.Thread(target=_ping, daemon=True, name="ws-keepalive")
        self._keepalive_thread = t  # 在 start() 前赋值，防止竞态
        t.start()

    def _stop_keepalive(self):
        """停止 keepalive 线程。"""
        if self._keepalive_thread is not None:
            self._keepalive_stop.set()
            self._keepalive_thread = None

    # ---- 连接管理 ----

    def ensure_connection(self):
        """
        确保 Chrome 已连接。

        委托给 xhs_snapshot.connect_chrome()，该函数会：
        - 优先复用已有 Chrome（PID 文件 + 端口检测）
        - 若无运行中实例则自动启动
        """
        if self._page is not None:
            try:
                self._page.run_cdp("Browser.getVersion")
                return self._page
            except Exception:
                self._log("Chrome 连接已断开，准备重连")
                self._page = None

        try:
            from src.xhs_snapshot import connect_chrome, CDP_PORT
        except ImportError:
            import sys as _sys
            from pathlib import Path as _Path
            _src_dir = str(_Path(__file__).resolve().parent)
            if _src_dir not in _sys.path:
                _sys.path.insert(0, _src_dir)
            from xhs_snapshot import connect_chrome, CDP_PORT

        port = getattr(self.config, 'cdp_port', None) or getattr(getattr(self.config, 'chrome', None), 'cdp_port', None) or CDP_PORT

        # 多账号：随机选取一个 Chrome profile
        try:
            from src.account_manager import AccountManager
        except ImportError:
            from account_manager import AccountManager
        acct_mgr = AccountManager(self.config)
        profile_path = acct_mgr.select_profile()
        self._log(f"连接 Chrome (端口 {port}, profile={profile_path})...")
        self._page = connect_chrome(port=port, auto_launch=True, profile_dir=profile_path)
        self._log("Chrome 连接成功")
        self._start_keepalive()
        return self._page

    def reconnect(self, old_page=None):
        """
        断开后重新连接。指数退避，最多重试 3 次。
        """
        self._stop_keepalive()

        # 保存断开前的页面上下文，便于重连后恢复
        saved_url = None
        saved_scroll = None
        if old_page is not None:
            try:
                saved_url = old_page.url
            except Exception:
                pass
            try:
                saved_scroll = int(old_page.run_js("return window.scrollY || 0"))
            except Exception:
                pass
            # 注意：不调用 old_page.quit()，因为这会关闭浏览器进程。
            # 我们只需释放引用让 CDP 连接自然断开，然后重新连接。

        self._reconnect_context = {'url': saved_url, 'scroll_y': saved_scroll}

        for attempt in range(1, 4):
            wait = 2 ** attempt  # 2, 4, 8 秒
            self._log(f"重连尝试 {attempt}/3 (等待 {wait}s)...")
            time.sleep(wait)

            try:
                page = self.ensure_connection()
                # 重新注入反检测 JS
                self._log("重新注入反检测 JS")
                try:
                    page.run_cdp(
                        'Page.addScriptToEvaluateOnNewDocument',
                        source=ANTI_DETECTION_JS
                    )
                except Exception:
                    pass
                return page
            except Exception as e:
                self._log(f"重连失败: {e}")

        raise RuntimeError("Chrome 重连失败，已重试 3 次")

    def cleanup_tabs(self, page=None):
        """
        关闭除当前页外的所有标签页，释放 Chrome 内存。
        每次采集结束后调用此方法可避免标签页累积。
        """
        p = page or self._page
        if p is None:
            return
        try:
            # 通过 CDP 获取所有目标
            target_info = p.run_cdp('Target.getTargetInfo')
            current_id = target_info.get('targetInfo', {}).get('targetId')
            all_targets = p.run_cdp('Target.getTargets').get('targetInfos', [])

            closed = 0
            for t in all_targets:
                tid = t.get('targetId')
                ttype = t.get('type', '')
                if tid and tid != current_id and ttype == 'page':
                    try:
                        p.run_cdp('Target.closeTarget', targetId=tid)
                        closed += 1
                    except Exception:
                        pass

            if closed > 0:
                self._log(f"已关闭 {closed} 个旧标签页")
        except Exception:
            pass

    def keep_alive(self):
        """检查 Chrome 是否存活的快捷方式。"""
        try:
            page = self.ensure_connection()
            page.run_cdp("Browser.getVersion")
            return True
        except Exception:
            return False

    def close(self):
        """
        清理 page 对象。
        注意：不关闭 Chrome 进程（让它继续运行供下次复用）。
        """
        self._stop_keepalive()
        if self._page is not None:
            try:
                self._page.quit()
            except Exception:
                pass
            self._page = None

    # ---- Session 过期检测 ----

    def detect_session_expiry(self, page=None) -> tuple:
        """
        检测当前 Session 是否过期。

        返回 (is_expired: bool, reason: str)。
        保守策略：宁可漏报也不误报。
        """
        p = page or self._page
        if p is None:
            return False, ""

        # 在当前页检测
        detector = SessionDetector(p)
        expired, reason = detector.check_all()
        if expired:
            self._log(f"⚠ Session 可能已过期: {reason}")
            return True, reason

        # 导航到首页做一次更彻底的检测
        self._log("导航到首页验证登录状态...")
        try:
            original_url = p.url
            p.get("https://www.xiaohongshu.com", timeout=15)
            time.sleep(2)
            expired, reason = SessionDetector(p).check_all()
            # 导航回原页面
            try:
                p.get(original_url, timeout=15)
            except Exception:
                pass
            if expired:
                self._log(f"⚠ Session 确认过期: {reason}")
                return True, reason
        except Exception as e:
            self._log(f"导航到首页验证登录时出错: {e}")

        return False, ""

    # ---- 通知 ----

    def notify_user(self, message: str, title: str = "xhs-scraper 通知"):
        """
        向用户发送通知。

        优先级：
          1. Windows MessageBox（模态弹窗，最显眼）
          2. Windows 提示音 + 控制台日志
          3. 写入 session_expired.flag 文件
        """
        log_msg = f"[通知] {title}: {message}"
        self._log(log_msg)

        # 写入 flag 文件（持久化记录）
        data_dir = (
            getattr(getattr(self.config, 'collection', None), 'data_dir', None)
            or getattr(self.config, 'data_dir', None)
            or 'data'
        )
        flag_path = os.path.join(data_dir, "session_expired.flag")
        try:
            os.makedirs(os.path.dirname(flag_path) or ".", exist_ok=True)
            with open(flag_path, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "title": title,
                    "message": message,
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Windows 平台：模态弹窗 + 提示音
        if sys.platform == "win32":
            try:
                import ctypes
                # 提示音
                ctypes.windll.kernel32.Beep(800, 300)
                ctypes.windll.kernel32.Beep(1000, 300)
                ctypes.windll.kernel32.Beep(1200, 400)
                # 模态弹窗
                ctypes.windll.user32.MessageBoxW(
                    0, message, title, 0x40 | 0x1000  # MB_ICONINFORMATION | MB_SYSTEMMODAL
                )
            except Exception:
                pass
        else:
            # 控制台 BEL 字符
            print("\a", end="", flush=True)

    # ---- Session flag 管理 ----

    def _flag_path(self) -> str:
        data_dir = (
            getattr(getattr(self.config, 'collection', None), 'data_dir', None)
            or getattr(self.config, 'data_dir', None)
            or 'data'
        )
        return os.path.join(data_dir, "session_expired.flag")

    def read_expiry_flag(self) -> dict | None:
        """读取 session_expired.flag，1 小时内的才视为有效。"""
        path = self._flag_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ts = data.get("timestamp", "")
            age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
            if age > 3600:
                return None
            return data
        except Exception:
            return None

    def clear_expiry_flag(self):
        """删除 session_expired.flag。"""
        try:
            path = self._flag_path()
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def _default_log(msg: str):
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [chrome] {msg}", flush=True)
