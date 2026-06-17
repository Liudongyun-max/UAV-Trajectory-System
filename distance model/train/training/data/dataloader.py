"""数据加载器封装"""
from torch.utils.data import DataLoader
from typing import Optional, Tuple

from .dataset import UAVTrajectoryDataset
from .transforms import FeatureTransform


def create_dataloaders(
    root_dir: str,
    train_ids: list,
    val_ids: list,
    test_ids: Optional[list] = None,
    batch_size: int = 64,
    seq_len: int = 20,
    future_steps: int = 5,
    input_dim: int = 8,
    negative_ratio: float = 0.3,
    num_workers: int = 4,
    pin_memory: bool = True,
    smooth: bool = False,
    smooth_window: int = 5,
    smooth_polyorder: int = 2,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    创建训练/验证/测试数据加载器
    
    Args:
        root_dir: 数据根目录
        train_ids: 训练集Episode ID
        val_ids: 验证集Episode ID
        test_ids: 测试集Episode ID (可选)
        batch_size: 批次大小
        seq_len: 序列长度
        future_steps: 未来步数
        input_dim: 输入特征维度
        negative_ratio: 负样本比例
        num_workers: 工作进程数
        pin_memory: 是否锁定内存
        smooth: 是否开启预平滑
        smooth_window: 平滑窗口
        smooth_polyorder: 平滑多阶次
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 实例化 transform
    train_transform = None
    if smooth:
        train_transform = FeatureTransform(
            smooth=smooth,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            add_noise=True,  # 训练集开启随机噪声作为数据增强
            noise_std=0.001,
        )
        
    val_transform = None
    if smooth:
        val_transform = FeatureTransform(
            smooth=smooth,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            add_noise=False,
        )

    train_loader = None
    if train_ids:
        train_dataset = UAVTrajectoryDataset(
            root_dir=root_dir,
            episode_ids=train_ids,
            seq_len=seq_len,
            future_steps=future_steps,
            input_dim=input_dim,
            negative_ratio=negative_ratio,
            normalize=True,
            transform=train_transform,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    
    val_loader = None
    if val_ids:
        val_dataset = UAVTrajectoryDataset(
            root_dir=root_dir,
            episode_ids=val_ids,
            seq_len=seq_len,
            future_steps=future_steps,
            input_dim=input_dim,
            negative_ratio=0,
            normalize=True,
            transform=val_transform,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    test_loader = None
    if test_ids:
        test_dataset = UAVTrajectoryDataset(
            root_dir=root_dir,
            episode_ids=test_ids,
            seq_len=seq_len,
            future_steps=future_steps,
            input_dim=input_dim,
            negative_ratio=0,
            normalize=True,
            transform=val_transform,  # 测试集使用 val_transform
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    return train_loader, val_loader, test_loader
