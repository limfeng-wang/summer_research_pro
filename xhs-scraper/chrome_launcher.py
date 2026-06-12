#!/usr/bin/env python3
import os
import sys
import time
import socket
import subprocess
import urllib.request
import shutil

CDP_PORT = 9222
XHS_URL = "https://www.xiaohongshu.com"
# 【关键改进】使用独立的隔离配置目录，绝不影响你日常使用的浏览器！
PROFILE_DIR = os.path.abspath("./profiles/chrome_main")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def find_chrome() -> str:
    """跨平台查找 Chrome 路径"""
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
        mac_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.exists(mac_path):
            return mac_path
    else:
        for name in ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']:
            path = shutil.which(name)
            if path:
                return path
    raise FileNotFoundError("无法在你的系统中找到 Google Chrome，请确认是否已安装或配置环境变量。")

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def wait_for_cdp(timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_port_in_use(CDP_PORT):
            try:
                r = urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/version", timeout=1)
                if r.status == 200:
                    return True
            except Exception:
                pass
        time.sleep(0.5)
    return False

def launch():
    try:
        chrome_path = find_chrome()
        log(f"找到 Chrome 路径: {chrome_path}")
    except Exception as e:
        log(f"❌ 错误: {e}")
        sys.exit(1)

    if is_port_in_use(CDP_PORT):
        log(f"❌ 错误: 端口 {CDP_PORT} 已经被占用！")
        log("请检查是否有之前的僵尸进程，或者修改 CDP_PORT。")
        sys.exit(1)

    os.makedirs(PROFILE_DIR, exist_ok=True)
    log(f"使用独立环境目录: {PROFILE_DIR}")

    # 启动命令组合
    cmd = [
        chrome_path,
        f"--user-data-dir={PROFILE_DIR}",
        f"--remote-debugging-port={CDP_PORT}",
        "--disable-blink-features=AutomationControlled",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-component-update",
        "--disable-breakpad",
        "--disable-domain-reliability",
        "--disable-client-side-phishing-detection",
        "--no-service-autorun",
        "--disable-features=ChromeWhatsNewUI,TranslateUI",
        "--window-size=1400,900",
        "--window-position=100,100",
        XHS_URL
    ]

    log("正在拉起 Chrome 浏览器...")
    try:
        # 使用 Popen 启动，不再使用 taskkill 杀掉全家
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"❌ 启动进程失败: {e}")
        sys.exit(1)

    log("等待调试端口就绪...")
    if wait_for_cdp():
        print("\n" + "="*50)
        log("Chrome CDP 启动成功!")
        log(f"调试端口: 127.0.0.1:{CDP_PORT}")
        log(f"隔离数据已保存在: {PROFILE_DIR}")
        print("="*50 + "\n")
        log("你现在可以运行爬虫脚本了。")
        
        try:
            print("请勿关闭此终端。按 Ctrl+C 退出并关闭该浏览器...")
            proc.wait()
        except KeyboardInterrupt:
            log("正在安全关闭 Chrome...")
            proc.terminate()
            proc.wait(timeout=3)
            log("已退出。")
    else:
        log("❌ 启动超时，未能连接到 CDP 端口。")
        proc.terminate()
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chrome CDP 启动器（小红书多账号支持）")
    parser.add_argument("--profile", default=None,
                        help="Chrome profile 目录，默认 ./profiles/chrome_main")
    parser.add_argument("--port", type=int, default=9222,
                        help="CDP 调试端口，默认 9222")
    args = parser.parse_args()
    if args.profile:
        PROFILE_DIR = os.path.abspath(args.profile)
    CDP_PORT = args.port
    launch()