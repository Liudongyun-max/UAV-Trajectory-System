#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

def process_final_distance(stable_pred_val, kalman_dist):
    event = None
    final_val = stable_pred_val
    
    if stable_pred_val < 0.0:
        event = "GRU_OUTPUT_INVALID"
        final_val = kalman_dist
        
    return final_val, event

def test_negative_distance_clamping_and_fallback():
    kalman_est = 15.6
    final_val, event = process_final_distance(-2.5, kalman_est)
    
    assert final_val == kalman_est
    assert event == "GRU_OUTPUT_INVALID"
    
    final_val, event = process_final_distance(12.3, kalman_est)
    assert final_val == 12.3
    assert event is None

    final_val, event = process_final_distance(0.0, kalman_est)
    assert final_val == 0.0
    assert event is None
