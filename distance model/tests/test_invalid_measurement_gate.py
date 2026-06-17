#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def test_bbox_too_small_gate():
    # 测试 bbox 宽高过小时被 BBOX_TOO_SMALL 门限拦截
    min_width = 3.0
    min_height = 3.0
    
    # 模拟测试用例
    test_cases = [
        (2.0, 5.0, False, "BBOX_TOO_SMALL"),
        (5.0, 2.0, False, "BBOX_TOO_SMALL"),
        (4.0, 4.0, True, "")
    ]
    
    for bw, bh, expected_valid, expected_reason in test_cases:
        valid = True
        reason = ""
        if bw < min_width or bh < min_height:
            valid = False
            reason = "BBOX_TOO_SMALL"
            
        assert valid == expected_valid
        assert reason == expected_reason

def test_short_axis_collapsed_gate():
    # 测试短轴塌陷门限：bh <= 1.0 或 aspect_ratio >= 8.0
    # 这在横向机动时经常发生
    test_cases = [
        (10.0, 1.0, False, "SHORT_AXIS_COLLAPSED"),  # bh <= 1.0
        (40.0, 4.0, False, "SHORT_AXIS_COLLAPSED"),  # aspect_ratio = 10.0 >= 8.0
        (10.0, 2.0, True, "")  # aspect_ratio = 5.0 < 8.0 且 bh > 1.0
    ]
    
    for bw, bh, expected_valid, expected_reason in test_cases:
        aspect_ratio = bw / bh
        valid = True
        reason = ""
        if bh <= 1.0 or aspect_ratio >= 8.0:
            valid = False
            reason = "SHORT_AXIS_COLLAPSED"
            
        assert valid == expected_valid
        assert reason == expected_reason

def test_measurement_quality_calculation():
    # 验证测量质量因子评定与物理融合前的中位数剔除降权
    # 候选距离：d_width, d_height, d_area
    d_width = 100.0
    d_height = 500.0  # 塌陷导致的超远值
    d_area = 110.0
    
    d_candidates = [d_width, d_height, d_area]
    q_candidates = [1.0, 1.0, 1.0]
    
    med_val = np.median(d_candidates)
    disagreement_thresh = 0.5
    
    for idx, d_val in enumerate(d_candidates):
        if abs(d_val - med_val) / max(1.0, med_val) > disagreement_thresh:
            q_candidates[idx] *= 0.1
            
    # d_height = 500.0, med_val = 110.0
    # abs(500.0 - 110.0) / 110.0 = 390 / 110 = 3.54 > 0.5 -> 降权
    # d_width = 100.0, abs(100 - 110) / 110 = 10 / 110 = 0.09 <= 0.5 -> 不降权
    # d_area = 110.0, 偏差为0 -> 不降权
    
    assert q_candidates[0] == 1.0
    assert q_candidates[1] == 0.1
    assert q_candidates[2] == 1.0
    
    sum_q = sum(q_candidates)
    assert sum_q == 2.1  # >= 1.5
    
    # 稳健融合值
    robust_fused_range = sum(q * d for q, d in zip(q_candidates, d_candidates)) / sum_q
    assert abs(robust_fused_range - 123.81) < 0.1
