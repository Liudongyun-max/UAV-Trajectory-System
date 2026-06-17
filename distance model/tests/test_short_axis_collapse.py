#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

def check_short_axis_collapse(bw, bh):
    aspect_ratio = bw / max(1e-6, bh)
    measurement_valid = True
    invalid_reason = ""
    
    if bh <= 1.0 or aspect_ratio >= 8.0:
        measurement_valid = False
        invalid_reason = "SHORT_AXIS_COLLAPSED"
        
    return measurement_valid, invalid_reason

def test_short_axis_collapse_conditions():
    valid, reason = check_short_axis_collapse(10.0, 1.0)
    assert valid is False
    assert reason == "SHORT_AXIS_COLLAPSED"
    
    valid, reason = check_short_axis_collapse(5.0, 0.8)
    assert valid is False
    assert reason == "SHORT_AXIS_COLLAPSED"

    valid, reason = check_short_axis_collapse(24.0, 3.0)
    assert valid is False
    assert reason == "SHORT_AXIS_COLLAPSED"
    
    valid, reason = check_short_axis_collapse(32.0, 3.0)
    assert valid is False
    assert reason == "SHORT_AXIS_COLLAPSED"
    
    valid, reason = check_short_axis_collapse(14.0, 3.0)
    assert valid is True
    assert reason == ""
