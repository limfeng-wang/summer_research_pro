@echo off
cd /d "%~dp0"
echo [%TIME%] 夜猫子采集调度器启动中...
echo 配置: 00:00-02:30 时段, 关键词: 牙痛/牙疼/口腔溃疡
echo.
python -m src.scheduler_daemon
pause
