@echo off
chcp 65001 >nul
title VoxCPM2 LoRA WebUI
echo ========================================
echo    VoxCPM2 LoRA 微调 WebUI - 启动中...
echo ========================================
echo.
echo 启动后请访问: http://localhost:7860
echo 关闭此窗口即可停止服务。
echo ========================================
echo.

cd /d "%~dp0"
start "" http://localhost:7860
"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe" "%~dp0lora_ft_webui.py"

echo.
echo 服务已停止。
pause
