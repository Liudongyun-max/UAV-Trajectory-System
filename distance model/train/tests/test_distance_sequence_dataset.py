#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pytest
import torch
from training.data.distance_sequence_dataset import DistanceSequenceDataset


def test_distance_sequence_dataset_shapes():
    # 构造假数据
    fold_dict = {
        "x": np.random.randn(10, 25, 16).astype(np.float32),
        "y": np.random.randn(10, 1).astype(np.float32),
        "missing_mask": np.zeros((10, 25)).astype(np.float32),
        "interpolated_mask": np.zeros((10, 25)).astype(np.float32),
        "meta": [{"nominal_distance": 50.0} for _ in range(10)]
    }
    
    ds = DistanceSequenceDataset(fold_dict)
    assert len(ds) == 10
    
    x_val, y_val, miss, interp, meta = ds[0]
    
    assert x_val.shape == (25, 16)
    assert y_val.shape == (1,)
    assert miss.shape == (25,)
    assert interp.shape == (25,)
    assert isinstance(meta, dict)
