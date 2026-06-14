@echo off
chcp 65001 >nul
title UAV 轨迹预测系统 - 环境配置脚本
color 0A

echo.
echo  ╔════════════════════════════════════════════════════════════╗
echo  ║        UAV 轨迹预测系统 - 环境自动配置脚本               ║
echo  ╚════════════════════════════════════════════════════════════╝
echo.

REM 检查 conda 是否可用
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 未检测到 conda！
    echo.
    echo 请先安装 Anaconda 或 Miniconda:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

echo [✓] 检测到 conda
echo.

REM 设置环境名称
set ENV_NAME=uav_gru

echo ========================================
echo   步骤 1/4: 创建 conda 环境
echo ========================================
echo 创建环境 '%ENV_NAME%' (Python 3.10)...
conda create -n %ENV_NAME% python=3.10 -y
if %errorlevel% neq 0 (
    echo [ERROR] 创建环境失败！
    pause
    exit /b 1
)
echo [✓] 环境创建成功
echo.

echo ========================================
echo   步骤 2/4: 安装依赖包
echo ========================================
echo 安装中，请稍候...
call conda activate %ENV_NAME%
pip install numpy pandas opencv-python torch scipy PyQt5 -q
if %errorlevel% neq 0 (
    echo [ERROR] 安装依赖失败！
    pause
    exit /b 1
)
echo [✓] 依赖安装成功
echo.

echo ========================================
echo   步骤 3/4: 验证安装
echo ========================================
python -c "import torch; import cv2; import PyQt5; import numpy; import pandas; from scipy.signal import savgol_filter; print('[✓] 所有依赖验证通过!')"
if %errorlevel% neq 0 (
    echo [ERROR] 验证失败！
    pause
    exit /b 1
)
echo.

echo ========================================
echo   步骤 4/4: 更新启动脚本
echo ========================================
echo 正在更新 启动UAV预测窗口.vbs...
(
echo Set shell = CreateObject^("WScript.Shell"^)
echo scriptDir = CreateObject^("Scripting.FileSystemObject"^).GetParentFolderName^(WScript.ScriptFullName^)
echo cmd = "cmd /c set KMP_DUPLICATE_LIB_OK=TRUE ^&^& set OMP_NUM_THREADS=1 ^&^& cd /d """ ^& scriptDir ^& """ ^&^& conda run -n %ENV_NAME% pythonw """ ^& scriptDir ^& "\predict_gui.py"""
echo shell.Run cmd, 0, False
) > "%~dp0启动UAV预测窗口.vbs"
echo [✓] 启动脚本已更新
echo.

echo.
echo  ╔════════════════════════════════════════════════════════════╗
echo  ║                    配置完成！                             ║
echo  ╚════════════════════════════════════════════════════════════╝
echo.
echo  启动方式:
echo    1. 双击 "启动UAV预测窗口.vbs" (推荐)
echo    2. 命令行: conda activate %ENV_NAME% ^&^& python predict_gui.py
echo.
pause
