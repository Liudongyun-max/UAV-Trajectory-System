"""数据集测试"""
import pytest
import torch
import numpy as np
from training.data.dataset import UAVTrajectoryDataset
from training.data.negative_generator import NegativeSampleGenerator


class TestNegativeSampleGenerator:
    """负样本生成器测试"""
    
    def test_generation(self):
        generator = NegativeSampleGenerator(seq_len=20, future_steps=5)
        samples = generator.generate(10)
        
        assert len(samples) == 10
        
        for sample in samples:
            assert "features" in sample
            assert "future_offsets" in sample
            assert "label" in sample
            
            assert sample["features"].shape == (20, 8)
            assert sample["future_offsets"].shape == (5, 2)
            assert sample["label"] == 0.0
    
    def test_random_walk(self):
        generator = NegativeSampleGenerator(seq_len=20, future_steps=5, seed=42)
        traj = generator._random_walk_trajectory()
        
        assert traj.shape == (25, 2)
        assert np.all(traj >= 0)
        assert np.all(traj <= 1)
    
    def test_oscillation(self):
        generator = NegativeSampleGenerator(seq_len=20, future_steps=5, seed=42)
        traj = generator._oscillation_trajectory()
        
        assert traj.shape == (25, 2)
        assert np.all(traj >= 0)
        assert np.all(traj <= 1)


class TestUAVTrajectoryDataset:
    """数据集测试"""
    
    def test_sample_structure(self):
        features = np.random.randn(20, 8).astype(np.float32)
        future_offsets = np.random.randn(5, 2).astype(np.float32)
        
        sample = {
            "features": features,
            "future_offsets": future_offsets,
            "label": 1.0,
        }
        
        assert sample["features"].shape == (20, 8)
        assert sample["future_offsets"].shape == (5, 2)
        assert sample["label"] == 1.0
