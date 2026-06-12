# Chrome CDP Launcher

$ErrorActionPreference = "Stop"

Write-Host "====================================================="
Write-Host " Chrome CDP Launcher"
Write-Host " Profile: XiaohongshuCDP (separate profile)"
Write-Host " Port: 9222"
Write-Host "====================================================="
Write-Host ""

$chromeExe = "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe"
$chromeProfile = "C:\Users\PC\AppData\Local\Google\Chrome\XiaohongshuCDP"
$targetUrl = "https://www.xiaohongshu.com"

if (-not (Test-Path $chromeExe)) {
    $chromeExe = "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe"
}

Write-Host "Starting Chrome ..."
Start-Process -FilePath $chromeExe -ArgumentList @(
    "--user-data-dir=$chromeProfile",
    "--remote-debugging-port=9222",
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
    $targetUrl
)

Write-Host "Waiting 10s for CDP..."
Start-Sleep -Seconds 10

Write-Host "====================================================="
Write-Host " Chrome is running with CDP on port 9222"
Write-Host " If needed, log in to xiaohongshu.com in the new window"
Write-Host ""
Write-Host " Then run in a new terminal:"
Write-Host "   python xhs_browse/safe_collector.py --keyword 牙痛 --max 3"
Write-Host "====================================================="
Write-Host ""
Read-Host "Press Enter to close this window (Chrome keeps running)"
