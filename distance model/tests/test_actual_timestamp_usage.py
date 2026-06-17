#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

def calculate_frame_delta(curr_msec, prev_msec, fps_val):
    curr_time_s = curr_msec / 1000.0
    dt = curr_time_s - (prev_msec / 1000.0)
    event = None
    
    if dt <= 0:
        dt = 1.0 / fps_val
        event = "TIME_GAP_DETECTED"
    elif dt > 2.0:
        event = "TIME_GAP_DETECTED"
        dt = 1.0 / fps_val
        
    return dt, event

def test_dynamic_timestamp_calculation():
    fps = 25.0
    
    dt, event = calculate_frame_delta(40.0, 0.0, fps)
    assert abs(dt - 0.04) < 1e-4
    assert event is None
    
    dt, event = calculate_frame_delta(85.0, 40.0, fps)
    assert abs(dt - 0.045) < 1e-4
    assert event is None

    dt, event = calculate_frame_delta(50.0, 50.0, fps)
    assert abs(dt - 0.04) < 1e-4
    assert event == "TIME_GAP_DETECTED"
    
    dt, event = calculate_frame_delta(3000.0, 50.0, fps)
    assert abs(dt - 0.04) < 1e-4
    assert event == "TIME_GAP_DETECTED"
