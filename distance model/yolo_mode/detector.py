#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
from pathlib import Path

class YOLOv5Detector:
    """
    YOLOv5 实时目标检测包装器类。
    内部自动将对外部 yolov5+GRU 路径的依赖内聚并动态导入 PyTorchImpl 目标定位器。
    """
    def __init__(self, model_path: str = "F:/UAV Trajectory System/distance model/yolo_mode/models/best include small.pt", conf_thres: float = 0.3, device: str = None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[YOLOv5Detector] 正在初始化。加载权重: {model_path}，运行设备: {self.device}")
        
        # 1. 动态挂载父目录中的 yolov5+GRU 引擎
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent  # UAV Trajectory System
        yolo_gru_dir = str(project_root / "yolov5+GRU")
        
        if yolo_gru_dir not in sys.path:
            sys.path.insert(0, yolo_gru_dir)
            
        # 2. 导入引擎实现
        try:
            from detector import PyTorchImpl
        except ImportError as e:
            print(f"[YOLOv5Detector] 错误：无法从 {yolo_gru_dir} 导入 detector 模块。请确认目录和文件存在！")
            raise e
            
        # 3. 实例化内部检测器
        self.detector = PyTorchImpl(model_path=model_path, conf_thres=conf_thres, device=self.device)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        前向推理。输入 BGR 图像帧，返回预测框 numpy.ndarray。
        格式: [ [x1, y1, x2, y2, confidence, class_id], ... ] (或由 PyTorchImpl 返回的格式)
        """
        return self.detector.detect(frame)
