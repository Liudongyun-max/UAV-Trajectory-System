@echo off
title UAV Trajectory Distance Visualizer UI
cd /d "%~dp0"
echo ===================================================
echo     Starting UAV Distance Visualizer UI...
echo     Environment: D:\anaconda3\envs\yolo
echo     Script: distance_model_gui.py
echo ===================================================
D:\anaconda3\envs\yolo\python.exe distance_model_gui.py
if %errorlevel% neq 0 (
    echo.
    echo GUI has exited with error code %errorlevel%.
    pause
)
