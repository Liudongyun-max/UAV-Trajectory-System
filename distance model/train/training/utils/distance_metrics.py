#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.abs(y_pred - y_true)
    
    # 帧间跳变 (相邻两帧预测值的差绝对值)
    diffs = np.abs(y_pred[1:] - y_pred[:-1])
    
    return {
        "mae_m": float(np.mean(err)),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "p95_abs_m": float(np.percentile(err, 95)) if len(err) > 0 else 0.0,
        "max_abs_m": float(np.max(err)) if len(err) > 0 else 0.0,
        "pred_std_m": float(np.std(y_pred)) if len(y_pred) > 0 else 0.0,
        "max_jump_m": float(np.max(diffs)) if len(diffs) > 0 else 0.0,
        "p95_jump_m": float(np.percentile(diffs, 95)) if len(diffs) > 0 else 0.0,
    }


class OneDimKalmanFilter:
    def __init__(self, q: float = 0.01, r: float = 0.5, p_init: float = 1.0):
        self.q = q  # 系统过程噪声
        self.r = r  # 测量噪声
        self.p = p_init
        self.x = None

    def update(self, z: float, is_missing: bool = False) -> float:
        if self.x is None:
            # 初始状态
            self.x = z
            return self.x
            
        # 1. 预测
        x_pred = self.x
        p_pred = self.p + self.q
        
        if is_missing:
            # 漏检时只外推, 不更新
            self.x = x_pred
            self.p = p_pred
        else:
            # 2. 测量更新
            k = p_pred / (p_pred + self.r)
            self.x = x_pred + k * (z - x_pred)
            self.p = (1.0 - k) * p_pred
            
        return self.x


def apply_kalman_filter(raw_preds: np.ndarray, missing_mask: np.ndarray, q: float = 0.01, r: float = 0.5) -> np.ndarray:
    kf = OneDimKalmanFilter(q=q, r=r)
    smoothed = []
    for val, is_miss in zip(raw_preds, missing_mask):
        smoothed.append(kf.update(val, is_missing=bool(is_miss)))
    return np.array(smoothed, dtype=np.float32)


def apply_ema(raw_preds: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    smoothed = []
    curr = None
    for val in raw_preds:
        if curr is None:
            curr = val
        else:
            curr = alpha * val + (1.0 - alpha) * curr
        smoothed.append(curr)
    return np.array(smoothed, dtype=np.float32)
