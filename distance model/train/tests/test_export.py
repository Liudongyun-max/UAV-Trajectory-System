"""导出测试"""
import pytest
import torch
from pathlib import Path
from training.models.gru_trajectory import UAVTrajectoryNet
from training.exporters.onnx_export import export_to_onnx
from training.exporters.torchscript_export import export_to_torchscript


class TestONNXExport:
    """ONNX导出测试"""
    
    def test_export(self, tmp_path):
        model = UAVTrajectoryNet(input_dim=8, hidden_dim=32, future_steps=5)
        model.eval()
        
        output_path = tmp_path / "model.onnx"
        
        export_to_onnx(model, str(output_path), input_dim=8, seq_len=20)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestTorchScriptExport:
    """TorchScript导出测试"""
    
    def test_trace_export(self, tmp_path):
        model = UAVTrajectoryNet(input_dim=8, hidden_dim=32, future_steps=5)
        model.eval()
        
        output_path = tmp_path / "model.pt"
        
        export_to_torchscript(model, str(output_path), input_dim=8, seq_len=20, method="trace")
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
