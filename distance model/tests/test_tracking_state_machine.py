#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

class StateMachineSimulator:
    def __init__(self):
        self.state = "LOST"
        self.consecutive_valid_frames = 0
        self.coasting_start_time = None
        self.reset_called = False
        self.reset_reasons = []
        
        self.reacquire_confirm_frames = 3
        self.gru_warmup_frames = 25
        self.lost_timeout_s = 1.0

    def reset_gru_state(self, reason):
        self.reset_called = True
        self.reset_reasons.append(reason)
        self.features_queue = []

    def update(self, measurement_valid, gate_decision, curr_time_s):
        prev_state = self.state
        
        if self.state == "LOST":
            if measurement_valid:
                self.consecutive_valid_frames += 1
                if self.consecutive_valid_frames >= self.reacquire_confirm_frames:
                    self.reset_gru_state("TARGET_REACQUIRED")
                    self.consecutive_valid_frames = 0
                    self.state = "WARMING_UP"
            else:
                self.consecutive_valid_frames = 0
                
        elif self.state == "WARMING_UP":
            if measurement_valid:
                self.consecutive_valid_frames += 1
                if self.consecutive_valid_frames >= self.gru_warmup_frames:
                    self.state = "TRACKING"
            else:
                self.consecutive_valid_frames = 0
                self.state = "COASTING"
                self.coasting_start_time = curr_time_s
                
        elif self.state in ["TRACKING", "DEGRADED"]:
            if measurement_valid:
                if gate_decision == "DOWNWEIGHT":
                    self.state = "DEGRADED"
                else:
                    self.state = "TRACKING"
            else:
                self.state = "COASTING"
                self.coasting_start_time = curr_time_s
                self.consecutive_valid_frames = 0
                
        elif self.state == "COASTING":
            if measurement_valid:
                self.consecutive_valid_frames += 1
                if self.consecutive_valid_frames >= self.reacquire_confirm_frames:
                    self.state = "TRACKING"
            else:
                self.consecutive_valid_frames = 0
                coast_elapsed = curr_time_s - self.coasting_start_time
                if coast_elapsed > self.lost_timeout_s:
                    self.state = "LOST"
                    self.reset_gru_state("TARGET_LOST")
                    
        return prev_state, self.state

def test_state_machine_transition_workflow():
    sm = StateMachineSimulator()
    assert sm.state == "LOST"
    
    sm.update(measurement_valid=True, gate_decision="ACCEPT", curr_time_s=0.0)
    assert sm.state == "LOST"
    sm.update(measurement_valid=True, gate_decision="ACCEPT", curr_time_s=0.04)
    assert sm.state == "LOST"
    sm.update(measurement_valid=True, gate_decision="ACCEPT", curr_time_s=0.08)
    assert sm.state == "WARMING_UP"
    assert sm.reset_called is True
    assert "TARGET_REACQUIRED" in sm.reset_reasons
    
    for frame in range(24):
        sm.update(measurement_valid=True, gate_decision="ACCEPT", curr_time_s=0.12 + frame*0.04)
    assert sm.state == "WARMING_UP"
    
    sm.update(measurement_valid=True, gate_decision="ACCEPT", curr_time_s=0.12 + 24*0.04)
    assert sm.state == "TRACKING"
    
    sm.update(measurement_valid=True, gate_decision="DOWNWEIGHT", curr_time_s=2.0)
    assert sm.state == "DEGRADED"
    
    sm.update(measurement_valid=True, gate_decision="ACCEPT", curr_time_s=2.04)
    assert sm.state == "TRACKING"
    
    sm.update(measurement_valid=False, gate_decision="REJECT", curr_time_s=2.08)
    assert sm.state == "COASTING"
    
    sm.update(measurement_valid=False, gate_decision="REJECT", curr_time_s=3.09)
    assert sm.state == "LOST"
    assert "TARGET_LOST" in sm.reset_reasons
