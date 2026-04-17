@echo off
chcp 65001 >nul
title VoxCPM2 语音合成系统
echo ========================================
echo    VoxCPM2 语音合成系统 - 启动中...
echo ========================================
echo.
echo 正在启动 Web Demo，请稍候...
echo 启动后会自动打开浏览器，如果没有自动打开，
echo 请手动访问: http://localhost:8808
echo.
echo 关闭此窗口即可停止服务。
echo ========================================
echo.

start "" http://localhost:8808

"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe" "%~dp0app.py" --port 8808

echo.
echo 服务已停止。
pause
