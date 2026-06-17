"""实验配置管理"""
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    """模型配置"""
    input_dim: int = 8
    hidden_dim: int = 32
    num_layers: int = 1
    future_steps: int = 5
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    cls_weight: float = 1.0
    reg_weight: float = 0.5
    grad_clip: float = 1.0
    patience: int = 15
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class DataConfig:
    """数据配置"""
    root_dir: str = "F:/UAV Trajectory System/export/input"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_by: str = "episode_id"
    negative_ratio: float = 0.3
    seq_len: int = 20
    normalize: bool = True
    smooth: bool = False
    smooth_window: int = 5
    smooth_polyorder: int = 2


@dataclass
class ExportConfig:
    """导出配置"""
    format: str = "pth"
    input_dim: int = 8
    seq_len: int = 20


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str = "gru_baseline"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "logs"
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        export_config = ExportConfig(**config_dict.get('export', {}))
        
        return cls(
            name=config_dict.get('name', 'gru_baseline'),
            model=model_config,
            training=training_config,
            data=data_config,
            export=export_config,
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cuda'),
            log_dir=config_dict.get('log_dir', 'logs'),
        )
    
    def to_yaml(self, path: str):
        """保存配置到YAML"""
        config_dict = {
            'name': self.name,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'export': self.export.__dict__,
            'seed': self.seed,
            'device': self.device,
            'log_dir': self.log_dir,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
