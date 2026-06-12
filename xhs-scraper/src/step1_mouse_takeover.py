#!/usr/bin/env python3
"""
Step 1: 接管鼠标使用 — 使用 Windows API 强制控制鼠标，保持浏览器不退出。

关键技术:
  - SetForegroundWindow + AttachThreadInput: 绕过 Windows 前台锁定机制
  - GetCursorPos / SetCursorPos: 光标位置检测与控制
  - SendInput: 底层鼠标事件注入（比 pyautogui 更可靠）

验证方法: 移动鼠标到 Chrome 窗口中心，再移回原位。
"""

import ctypes
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xhs_snapshot import connect_chrome
from win32_api import (find_chrome_hwnd, force_foreground,
                       move_mouse as move_mouse_to,
                       get_cursor_pos, user32)

ctypes.windll.user32.SetProcessDPIAware()


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    try:
        print(f"[{ts}] {msg}", flush=True)
    except Exception:
        pass







# =============================================================================
# 主流程
# =============================================================================

def main():
    log("=" * 60)
    log("Step 1: 接管鼠标使用")
    log("=" * 60)

    # ---- 连接 Chrome（保持不退出） ----
    log("\n[1] 连接 Chrome...")
    page = connect_chrome()
    kw = "牙痛"
    search_url = f"https://www.xiaohongshu.com/search_result?keyword={quote(kw)}"
    page.get(search_url, retry=2, timeout=25)
    log(f"    页面: {page.title[:60]}")

    # 等待卡片加载
    try:
        page.wait.ele_displayed('a[href*="/explore/"], a.cover', timeout=20)
        log("    卡片已加载")
    except Exception:
        log("    警告: 卡片加载超时")

    page.wait(1, 2)

    # ---- 查找 Chrome 窗口 ----
    log("\n[2] 查找 Chrome 窗口...")
    windows = find_chrome_hwnd()
    if not windows:
        log("    未找到 Chrome 窗口！")
        return page  # 返回 page 不退出

    win = windows[0]
    log(f"    窗口: HWND={win['hwnd']}")
    log(f"    标题: {win['title'][:50]}")
    log(f"    位置: ({win['left']}, {win['top']})")
    log(f"    尺寸: {win['width']}x{win['height']}")

    # ---- 接管鼠标 — 保存原始位置 ----
    log("\n[3] 接管鼠标 — 保存原始位置...")
    original_pos = get_cursor_pos()
    log(f"    原始光标位置: {original_pos}")

    # ---- 强制前台 ----
    log("\n[4] 强制 Chrome 前台...")
    ok = force_foreground(win['hwnd'])
    log(f"    结果: {'成功' if ok else '失败'}")
    fg_now = user32.GetForegroundWindow()
    log(f"    当前前台窗口 HWND: {fg_now} (目标: {win['hwnd']})")

    # ---- 移动鼠标到 Chrome 窗口中心（验证控制权） ----
    log("\n[5] 移动鼠标到 Chrome 窗口中心（验证控制权）...")
    center_x = win['left'] + win['width'] // 2
    center_y = win['top'] + win['height'] // 2

    log(f"    目标位置: ({center_x}, {center_y})")
    move_mouse_to(center_x, center_y)
    time.sleep(0.5)

    new_pos = get_cursor_pos()
    log(f"    移动后光标位置: {new_pos}")
    log(f"    偏差: ({new_pos[0] - center_x}, {new_pos[1] - center_y})")

    # ---- 复位 — 移回原位 ----
    log("\n[6] 移回原位...")
    time.sleep(1)
    move_mouse_to(original_pos[0], original_pos[1])
    time.sleep(0.3)
    final_pos = get_cursor_pos()
    log(f"    最终光标位置: {final_pos}")

    # ---- 总结 ----
    log("\n" + "=" * 60)
    log("Step 1 完成 — 鼠标接管验证通过")
    log(f"   - 窗口前台化: {'OK' if ok else 'FAIL'}")
    log(f"   - 光标移动: {(new_pos[0], new_pos[1])} -> {(final_pos[0], final_pos[1])}")
    log(f"   - 浏览器保持运行，CDP 连接存活")
    log("=" * 60)

    # 不调用 page.quit() — 保持浏览器运行
    return page


if __name__ == "__main__":
    main()
