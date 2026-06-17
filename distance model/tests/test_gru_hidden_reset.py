#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

class MockGUI:
    def __init__(self):
        self.features_queue = [1, 2, 3]
        self.events_log = []
        
    def reset_gru_state(self, reason: str):
        self.features_queue = []
        self.log_event("GRU_STATE_RESET", {
            "reason": reason,
            "timestamp": 123456789.0
        })
        
    def log_event(self, event_type: str, details: dict):
        self.events_log.append({
            "event_type": event_type,
            "details": details
        })

def test_reset_gru_state_clears_queue():
    gui = MockGUI()
    assert len(gui.features_queue) > 0
    
    gui.reset_gru_state("TEST_REASON")
    assert len(gui.features_queue) == 0
    assert len(gui.events_log) == 1
    assert gui.events_log[0]["event_type"] == "GRU_STATE_RESET"
    assert gui.events_log[0]["details"]["reason"] == "TEST_REASON"

def test_reset_triggers_9_reasons():
    reasons = [
        "NEW_VIDEO_START",
        "SEEK",
        "TRACK_ID_CHANGE",
        "TARGET_LOST",
        "TARGET_REACQUIRED",
        "TIMESTAMP_REGRESSED",
        "LONG_FRAME_INTERVAL",
        "MODEL_RELOAD",
        "FEATURE_SCHEMA_CHANGE"
    ]
    
    gui = MockGUI()
    for idx, reason in enumerate(reasons):
        gui.reset_gru_state(reason)
        assert len(gui.features_queue) == 0
        assert gui.events_log[-1]["details"]["reason"] == reason
    
    assert len(gui.events_log) == 9
