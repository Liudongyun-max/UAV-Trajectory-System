#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

class TargetTrackerSimulator:
    def __init__(self):
        self.state = "LOST"
        self.consecutive_valid_frames = 0
        self.features_queue = []
        
        self.reacquire_confirm_frames = 3
        self.gru_warmup_frames = 25

    def reset_gru_state(self, reason):
        self.features_queue = []
        
    def add_feature(self, feat):
        self.features_queue.append(feat)
        if len(self.features_queue) > 25:
            self.features_queue.pop(0)

    def process_frame(self, measurement_valid, feat_vector):
        if self.state == "LOST":
            if measurement_valid:
                self.consecutive_valid_frames += 1
                if self.consecutive_valid_frames >= self.reacquire_confirm_frames:
                    self.state = "WARMING_UP"
                    self.reset_gru_state("TARGET_REACQUIRED")
                    self.consecutive_valid_frames = 0
            else:
                self.consecutive_valid_frames = 0
                
        elif self.state == "WARMING_UP":
            if measurement_valid:
                self.add_feature(feat_vector)
                self.consecutive_valid_frames += 1
                if self.consecutive_valid_frames >= self.gru_warmup_frames:
                    self.state = "TRACKING"
            else:
                self.state = "LOST"
                self.reset_gru_state("TARGET_LOST")
                self.consecutive_valid_frames = 0
                
        elif self.state == "TRACKING":
            if measurement_valid:
                self.add_feature(feat_vector)
            else:
                self.state = "LOST"
                self.reset_gru_state("TARGET_LOST")
                
        if self.state == "WARMING_UP":
            return "MLP_FALLBACK", len(self.features_queue)
        elif self.state == "TRACKING":
            return "GRU_STABLE", len(self.features_queue)
        else:
            return "N/A", len(self.features_queue)

def test_reacquisition_warmup_pipeline():
    tracker = TargetTrackerSimulator()
    assert tracker.state == "LOST"
    
    # Frame 0 & 1: not reacquired yet, remains LOST
    out_mode, queue_len = tracker.process_frame(True, [0])
    assert out_mode == "N/A"
    out_mode, queue_len = tracker.process_frame(True, [1])
    assert out_mode == "N/A"
    
    # Frame 2: 3rd consecutive valid frame, transition to WARMING_UP
    out_mode, queue_len = tracker.process_frame(True, [2])
    assert tracker.state == "WARMING_UP"
    assert out_mode == "MLP_FALLBACK"
    assert queue_len == 0
    
    # Needs another 24 valid frames to complete warming up (total 25 frames)
    for i in range(1, 25):
        out_mode, queue_len = tracker.process_frame(True, [i])
        assert out_mode == "MLP_FALLBACK"
        assert queue_len == i
        
    out_mode, queue_len = tracker.process_frame(True, [25])
    assert tracker.state == "TRACKING"
    assert out_mode == "GRU_STABLE"
    assert queue_len == 25
