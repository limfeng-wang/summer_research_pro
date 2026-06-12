@echo off
chcp 65001 >nul
title Chrome CDP Launcher (Dedicated Profile)

echo =====================================================
echo  Chrome CDP Launcher
echo  Profile: XiaohongshuCDP
echo  Port: 9222
echo =====================================================
echo.
echo  [Note: Your existing Chrome will NOT be closed]
echo.
echo [Step 1/2] Starting Chrome with dedicated profile...
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" ^
    --user-data-dir="C:\Users\PC\AppData\Local\Google\Chrome\XiaohongshuCDP" ^
    --remote-debugging-port=9222 ^
    --disable-blink-features=AutomationControlled ^
    --no-first-run --no-default-browser-check ^
    --disable-background-networking ^
    --disable-sync ^
    --disable-component-update ^
    --disable-breakpad ^
    --disable-domain-reliability ^
    --disable-client-side-phishing-detection ^
    --no-service-autorun ^
    --disable-features=ChromeWhatsNewUI,TranslateUI ^
    --window-size=1400,900 --window-position=100,100 ^
    "https://www.xiaohongshu.com"

echo [Step 2/2] Waiting for CDP port...
timeout /t 8 /nobreak >nul

echo.
echo  Chrome should now be open with Xiaohongshu.com
echo  If not logged in, please log in (one-time only)
echo.
echo  Now run: python xhs_browse/safe_collector.py
echo.
pause
