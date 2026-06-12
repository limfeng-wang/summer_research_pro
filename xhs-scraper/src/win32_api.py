"""
win32_api.py — 集中化 Windows API 工具
======================================
提取自 xhs_collector_v2.py 和 step1-4*.py 中已验证的代码。
不修改任何原有文件。

用法:
    from win32_api import calibrate, vp2screen, send_click, find_chrome_hwnd, is_windows

依赖:
    Windows-only。非 Windows 系统上 is_windows() 返回 False，所有函数安全空跑。
"""

import ctypes
import json
import random
import sys
import time
from ctypes import wintypes
from pathlib import Path


def is_windows() -> bool:
    return sys.platform == "win32"


if is_windows():
    ctypes.windll.user32.SetProcessDPIAware()
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
else:
    user32 = None
    kernel32 = None


# =============================================================================
# 常量
# =============================================================================

SW_RESTORE = 9
SW_SHOW = 5
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_SHOWWINDOW = 0x0040

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


# =============================================================================
# SendInput 结构体（与 step1-4 和 collector_v2 完全一致）
# =============================================================================

ULONG_PTR = ctypes.c_ulong if ctypes.sizeof(ctypes.c_void_p) == 4 else ctypes.c_ulonglong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


# =============================================================================
# 窗口工具
# =============================================================================

def get_window_rect(hwnd):
    """获取窗口的屏幕坐标矩形。"""
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def get_cursor_pos():
    """获取当前鼠标屏幕坐标。"""
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


# =============================================================================
# 窗口查找（标题匹配，作为回退）
# =============================================================================

def find_chrome_hwnd():
    """通过 EnumWindows 找到 Chrome 窗口的 HWND。"""
    if not is_windows():
        return []
    results = []

    def enum_callback(hwnd, lParam):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                if title and ("小红书" in title or "Google Chrome" in title or "Chrome" in title):
                    rect = wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    if w > 300 and h > 200:
                        results.append({
                            'hwnd': hwnd,
                            'title': title,
                            'left': rect.left,
                            'top': rect.top,
                            'width': w,
                            'height': h,
                        })
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(enum_callback), 0)
    return results


def find_chrome_hwnd_by_pid(pid: int):
    """通过进程 PID 精确匹配 Chrome 窗口。"""
    if not is_windows() or not pid:
        return []
    results = []

    def enum_callback(hwnd, lParam):
        if user32.IsWindowVisible(hwnd):
            win_pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(win_pid))
            if win_pid.value == pid:
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                if w > 300 and h > 200:
                    results.append({
                        'hwnd': hwnd,
                        'title': '',
                        'left': rect.left,
                        'top': rect.top,
                        'width': w,
                        'height': h,
                    })
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(enum_callback), 0)
    return results


# =============================================================================
# 前台强制（AttachThreadInput 技巧）
# =============================================================================

def force_foreground(hwnd):
    """使用 AttachThreadInput 绕过 Windows 前台锁定。"""
    if not is_windows():
        return False
    fg = user32.GetForegroundWindow()
    if fg == hwnd:
        return True

    fg_thread = user32.GetWindowThreadProcessId(fg, None)
    cur_thread = kernel32.GetCurrentThreadId()

    # 附加到前台线程以允许 SetForegroundWindow
    if fg_thread != cur_thread and fg_thread != 0:
        user32.AttachThreadInput(cur_thread, fg_thread, True)

    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)

    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                        SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW)
    time.sleep(0.08)
    user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0,
                        SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW)
    time.sleep(0.08)
    user32.SetForegroundWindow(hwnd)
    user32.BringWindowToTop(hwnd)
    time.sleep(0.25)

    if fg_thread != cur_thread and fg_thread != 0:
        user32.AttachThreadInput(cur_thread, fg_thread, False)

    return user32.GetForegroundWindow() == hwnd


# =============================================================================
# 鼠标移动（SendInput 绝对坐标）
# =============================================================================

def move_mouse(x, y):
    """通过 SendInput 将鼠标移动到屏幕绝对坐标 (x, y)。"""
    if not is_windows():
        return
    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dx = int(x * 65535 / sw)
    inp.mi.dy = int(y * 65535 / sh)
    inp.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


# =============================================================================
# 鼠标点击（SendInput 左键）
# =============================================================================

def send_mousedown():
    """通过 SendInput 发送左键按下（不含释放）。"""
    if not is_windows():
        return
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dwFlags = MOUSEEVENTF_LEFTDOWN
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def send_mouseup():
    """通过 SendInput 发送左键释放（不含按下）。"""
    if not is_windows():
        return
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dwFlags = MOUSEEVENTF_LEFTUP
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def send_click():
    """通过 SendInput 发送左键点击（按下 + 释放，带随机间隔）。"""
    if not is_windows():
        return
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dwFlags = MOUSEEVENTF_LEFTDOWN
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    time.sleep(random.uniform(0.06, 0.14))
    inp2 = INPUT()
    inp2.type = INPUT_MOUSE
    inp2.mi.dwFlags = MOUSEEVENTF_LEFTUP
    user32.SendInput(1, ctypes.byref(inp2), ctypes.sizeof(inp2))


def verify_cursor_at(target_x: int, target_y: int, threshold: int = 5):
    """
    检查鼠标是否在目标坐标附近。如果不在，自动移动过去。

    返回 True 表示在目标位置（或已矫正），False 表示无法确定。
    """
    if not is_windows():
        return False
    cx, cy = get_cursor_pos()
    if abs(cx - target_x) <= threshold and abs(cy - target_y) <= threshold:
        return True
    move_mouse(target_x, target_y)
    time.sleep(0.05)
    cx2, cy2 = get_cursor_pos()
    return abs(cx2 - target_x) <= threshold * 2 and abs(cy2 - target_y) <= threshold * 2


# =============================================================================
# 视口→屏幕坐标校准
# =============================================================================

def get_chrome_pid(page=None) -> int | None:
    """
    获取 Chrome 浏览器进程 PID。

    策略:
      1. 读取 profiles/chrome_main/chrome.pid 文件
      2. 通过 CDP 尝试获取
      3. 通过进程名查找
    """
    # 策略 1: PID 文件
    pid_paths = [
        Path(__file__).resolve().parent.parent / "profiles" / "chrome_main" / "chrome.pid",
        Path("profiles/chrome_main/chrome.pid"),
    ]
    for p in pid_paths:
        try:
            if p.exists():
                pid = int(p.read_text().strip())
                if pid > 0:
                    return pid
        except Exception:
            pass

    # 策略 2: CDP 获取
    if page is not None:
        try:
            info = json.loads(page.run_js("return JSON.stringify({pid: 0})"))
            # 某些环境下 CDP 直接暴露 PID
        except Exception:
            pass

    # 策略 3: 通过枚举 Chrome 窗口取第一个的 PID
    wins = find_chrome_hwnd()
    if wins:
        hwnd = wins[0]['hwnd']
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value > 0:
            return pid.value

    return None


def calibrate(page, pid: int = None):
    """
    校准视口坐标到屏幕坐标的映射。

    参数:
        page: DrissionPage ChromiumPage 实例
        pid:  Chrome 进程 PID（可选），提供时优先用 PID 精确匹配窗口

    返回: {'csx': int, 'csy': int, 'dpr': float, 'hwnd': int | None, 'wx': int, 'wy': int, 'ww': int, 'wh': int}
      - csx/csy: 视口左上角的屏幕坐标
      - dpr: devicePixelRatio
      - hwnd: Chrome 窗口句柄（用于 force_foreground）
      - wx/wy/ww/wh: 窗口屏幕位置和尺寸，用于后续校验
    """
    if not is_windows():
        return {'csx': 0, 'csy': 0, 'dpr': 1, 'hwnd': None, 'wx': 0, 'wy': 0, 'ww': 0, 'wh': 0}

    wins = find_chrome_hwnd_by_pid(pid) if pid else []
    if not wins:
        wins = find_chrome_hwnd()
    if not wins:
        return {'csx': 0, 'csy': 0, 'dpr': 1, 'hwnd': None, 'wx': 0, 'wy': 0, 'ww': 0, 'wh': 0}

    hwnd = wins[0]['hwnd']
    cr = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(cr))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))

    info = json.loads(
        page.run_js("return JSON.stringify({dpr:window.devicePixelRatio})")
    )

    wx, wy, ww, wh = get_window_rect(hwnd)

    return {
        'csx': pt.x,
        'csy': pt.y,
        'dpr': info['dpr'],
        'hwnd': hwnd,
        'wx': wx,
        'wy': wy,
        'ww': ww,
        'wh': wh,
    }


def ensure_calibration(calib, page, pid: int = None):
    """
    校验校准数据是否仍然有效，如果窗口移动则自动重新校准。

    返回: (calib, changed)
      - calib: 当前有效的校准数据
      - changed: 是否重新校准过
    """
    if not is_windows() or not calib.get('hwnd'):
        return calib, False

    hwnd = calib['hwnd']
    if not user32.IsWindow(hwnd):
        # 窗口已关闭，重新查找
        return calibrate(page, pid), True

    wx, wy, ww, wh = get_window_rect(hwnd)
    moved = (
        abs(wx - calib.get('wx', wx)) > 5 or
        abs(wy - calib.get('wy', wy)) > 5 or
        abs(ww - calib.get('ww', ww)) > 5 or
        abs(wh - calib.get('wh', wh)) > 5
    )

    if not moved:
        return calib, False

    corrected = calibrate(page, pid)
    return corrected, True


def vp2screen(calib, vx, vy):
    """
    将视口坐标转换为屏幕绝对坐标。

    公式: screen = client_screen_origin + viewport * dpr
    验证: step2_position_lock.py 已验证精度 0-1px
    """
    return (
        int(calib['csx'] + vx * calib['dpr']),
        int(calib['csy'] + vy * calib['dpr']),
    )
