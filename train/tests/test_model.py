"""模型测试"""
import pytest
import torch
from training.models.gru_trajectory import UAVTrajectoryNet


class TestUAVTrajectoryNet:
    """GRU模型测试"""
    
    def test_forward(self):
        model = UAVTrajectoryNet(input_dim=8, hidden_dim=32, future_steps=5)
        x = torch.randn(4, 20, 8)
        
        logits, future_offsets = model(x)
        
        assert logits.shape == (4, 1)
        assert future_offsets.shape == (4, 5, 2)
    
    def test_predict_proba(self):
        model = UAVTrajectoryNet(input_dim=8, hidden_dim=32, future_steps=5)
        x = torch.randn(4, 20, 8)
        
        proba = model.predict_proba(x)
        
        assert proba.shape == (4, 1)
        assert torch.all(proba >= 0)
        assert torch.all(proba <= 1)
    
    def test_predict_with_confidence(self):
        model = UAVTrajectoryNet(input_dim=8, hidden_dim=32, future_steps=5)
        x = torch.randn(4, 20, 8)
        
        result = model.predict_with_confidence(x, threshold=0.5)
        
        assert "logits" in result
        assert "proba" in result
        assert "predictions" in result
        assert "future_offsets" in result
        
        assert result["predictions"].shape == (4, 1)
    
    def test_get_config(self):
        model = UAVTrajectoryNet(input_dim=8, hidden_dim=32, num_layers=1, future_steps=5, dropout=0.1)
        config = model.get_config()
        
        assert config["input_dim"] == 8
        assert config["hidden_dim"] == 32
        assert config["num_layers"] == 1
        assert config["future_steps"] == 5
        assert config["dropout"] == 0.1
    
    def test_count_parameters(self):
        model = UAVTrajectoryNet(input_dim=8, hidden_dim=32, future_steps=5)
        n_params = model.count_parameters()
        
        assert n_params > 0
        assert isinstance(n_params, int)
    
    def test_invalid_input(self):
        model = UAVTrajectoryNet(input_dim=8, hidden_dim=32, future_steps=5)
        
        with pytest.raises(ValueError):
            x = torch.randn(4, 20)
            model(x)
    
    def test_save_load(self, tmp_path):
        model = UAVTrajectoryNet(input_dim=8, hidden_dim=32, future_steps=5)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
        }
        
        path = tmp_path / "test_model.pth"
        torch.save(checkpoint, path)
        
        loaded_model = UAVTrajectoryNet.load_from_checkpoint(str(path))
        
        x = torch.randn(4, 20, 8)
        
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            logits1, offsets1 = model(x)
            logits2, offsets2 = loaded_model(x)
        
        assert torch.allclose(logits1, logits2)
        assert torch.allclose(offsets1, offsets2)
