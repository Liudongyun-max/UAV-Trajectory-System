#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

def run_fusion_logic(d_candidates, q_candidates, disagreement_thresh=0.5, solidity=None, min_solidity=0.85):
    med_val = np.median(d_candidates)
    q_out = list(q_candidates)
    
    for idx, d_val in enumerate(d_candidates):
        if abs(d_val - med_val) / max(1.0, med_val) > disagreement_thresh:
            q_out[idx] *= 0.1
            
    if solidity is not None and solidity < min_solidity:
        for idx in range(len(q_out)):
            q_out[idx] *= 0.5
            
    sum_q = sum(q_out)
    valid = sum_q >= 1.5
    
    if valid:
        fused = sum(q * d for q, d in zip(q_out, d_candidates)) / sum_q
    else:
        fused = sum(d_candidates) / len(d_candidates)
        
    return valid, fused, q_out

def test_adaptive_robust_fusion_normal():
    d_candidates = [10.0, 10.5, 10.2]
    q_candidates = [1.0, 1.0, 1.0]
    valid, fused, q_out = run_fusion_logic(d_candidates, q_candidates, disagreement_thresh=0.5)
    
    assert valid is True
    assert q_out == [1.0, 1.0, 1.0]
    assert abs(fused - 10.2333) < 1e-3

def test_adaptive_robust_fusion_with_outlier():
    d_candidates = [10.0, 10.2, 25.0]
    q_candidates = [1.0, 1.0, 1.0]
    valid, fused, q_out = run_fusion_logic(d_candidates, q_candidates, disagreement_thresh=0.5)
    
    assert valid is True
    assert q_out == [1.0, 1.0, 0.1]
    assert abs(fused - (10.0 + 10.2 + 2.5) / 2.1) < 1e-4

def test_adaptive_robust_fusion_invalid_weight():
    d_candidates = [10.0, 25.0, 50.0]
    q_candidates = [1.0, 1.0, 1.0]
    valid, fused, q_out = run_fusion_logic(d_candidates, q_candidates, disagreement_thresh=0.5)
    
    assert valid is False
    assert q_out == [0.1, 1.0, 0.1]

def test_adaptive_robust_fusion_solidity_degrade():
    d_candidates = [10.0, 10.5, 10.2]
    q_candidates = [1.0, 1.0, 1.0]
    valid, fused, q_out = run_fusion_logic(d_candidates, q_candidates, disagreement_thresh=0.5, solidity=0.80, min_solidity=0.85)
    
    assert q_out == [0.5, 0.5, 0.5]
    assert sum(q_out) == 1.5
    assert valid is True
