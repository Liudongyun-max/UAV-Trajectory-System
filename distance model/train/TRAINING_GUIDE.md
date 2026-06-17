# UAV Trajectory Model Training Guide

## 1. Overview

This document provides a comprehensive guide for training the UAV trajectory recognition model. The model is designed to:
- Classify whether a trajectory belongs to a real UAV or background noise
- Predict future 5-frame positions from 20-frame history

## 2. Environment Setup

### 2.1 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- NVIDIA GPU drivers (if using CUDA)

### 2.2 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2.3 Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

## 3. Data Preparation

### 3.1 Data Structure

The training data should be organized as:

```
export/input/
├── run001/
│   ├── hover/
│   │   ├── episode_hover_d300_vhover_empty_clean_seed0001/
│   │   │   ├── ideal_tracks.csv
│   │   │   ├── measured_tracks.csv
│   │   │   └── truth.csv
│   │   └── ...
│   ├── approaching/
│   └── ...
└── run002/
    └── hover/
        └── ...
```

### 3.2 CSV Format

**ideal_tracks.csv** (Ground Truth):
```csv
frame_id,x_center,y_center,bbox_width,bbox_height,visible,track_id
0,1280.5,720.3,45.2,15.8,True,1
1,1281.2,720.1,45.0,15.7,True,1
...
```

**measured_tracks.csv** (Noisy Detection):
```csv
frame_id,x_center,y_center,bbox_width,bbox_height,detector_confidence,is_missing,track_id
0,1282.1,719.8,44.8,16.1,0.85,False,1
1,1280.9,721.5,45.5,15.2,0.78,False,1
...
```

## 4. Training

### 4.1 Basic Training

```bash
cd train
python scripts/train.py --config configs/gru_baseline.yaml
```

### 4.2 Training with Custom Settings

```bash
python scripts/train.py \
    --config configs/gru_baseline.yaml \
    --device cuda \
    --epochs 200 \
    --batch_size 32
```

### 4.3 Training Configuration

Edit `configs/gru_baseline.yaml`:

```yaml
name: gru_baseline

model:
  input_dim: 8
  hidden_dim: 32
  num_layers: 1
  future_steps: 5
  dropout: 0.1

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  cls_weight: 1.0
  reg_weight: 0.5
  grad_clip: 1.0
  patience: 15
```

### 4.4 Monitoring Training

```bash
# TensorBoard
tensorboard --logdir logs/gru_baseline
```

## 5. Evaluation

### 5.1 Run Evaluation

```bash
python scripts/evaluate.py \
    --model models/gru_baseline_final.pth \
    --data_dir export/input \
    --split test \
    --visualize
```

### 5.2 Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy | Overall classification accuracy | > 0.95 |
| Precision | True positive rate | > 0.95 |
| Recall | Sensitivity | > 0.95 |
| F1 | Harmonic mean of precision and recall | > 0.95 |
| ADE | Average displacement error | < 5px |
| FDE | Final displacement error | < 10px |

## 6. Model Export

### 6.1 Export to ONNX

```bash
python scripts/export_model.py \
    --model models/gru_baseline_final.pth \
    --format onnx \
    --output models/uav_trajectory.onnx
```

### 6.2 Export to TorchScript

```bash
python scripts/export_model.py \
    --model models/gru_baseline_final.pth \
    --format torchscript \
    --output models/uav_trajectory.pt
```

### 6.3 Deploy to RK3588

1. Export to ONNX
2. Convert to RKNN format using rknn-toolkit2
3. Load in main.py

## 7. Model Architecture

```
Input: [Batch, 20, 8]
    │
    ▼
GRU (input=8, hidden=32, layers=1)
    │
    ▼
Hidden: [Batch, 32]
    │
    ├──▶ Classifier: Linear(32→16) → ReLU → Linear(16→1) → Logit
    │
    └──▶ Predictor: Linear(32→32) → ReLU → Linear(32→10) → [Batch, 5, 2]
```

### 7.1 Input Features

| Feature | Description |
|---------|-------------|
| x_norm | Normalized x position (x/2560) |
| y_norm | Normalized y position (y/1440) |
| relative_x | Cumulative x displacement from first frame |
| relative_y | Cumulative y displacement from first frame |
| velocity_x | Frame-to-frame x velocity |
| velocity_y | Frame-to-frame y velocity |
| acceleration_x | Frame-to-frame x acceleration |
| acceleration_y | Frame-to-frame y acceleration |

### 7.2 Output

- **Classification logit**: [B, 1] → sigmoid → UAV probability (0.0~1.0)
- **Future offsets**: [B, 5, 2] → relative position offsets for next 5 frames

## 8. Negative Sample Generation

The training process automatically generates negative samples:

| Type | Description |
|------|-------------|
| Random Walk | No net displacement, random steps |
| Oscillation | High-frequency back-and-forth |
| Static Point | Fixed position with small jitter |
| Bird Trajectory | Smooth but different from UAV |

## 9. Troubleshooting

### 9.1 CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train.py --config configs/gru_baseline.yaml --batch_size 16
```

### 9.2 Slow Training

```bash
# Reduce num_workers if CPU-bound
python scripts/train.py --config configs/gru_baseline.yaml --batch_size 32

# Use mixed precision (requires PyTorch 1.6+)
# Add to training loop:
# scaler = torch.cuda.amp.GradScaler()
```

### 9.3 Overfitting

- Increase `negative_ratio` in config
- Add more dropout
- Use data augmentation
- Early stopping is enabled by default

## 10. Project Structure

```
train/
├── training/
│   ├── config/          # Configuration management
│   ├── data/            # Dataset and dataloader
│   ├── models/          # Model definitions
│   ├── losses/          # Loss functions
│   ├── trainers/        # Training loops
│   ├── evaluators/      # Evaluation metrics
│   ├── exporters/       # Model export
│   └── utils/           # Utility functions
├── scripts/             # Entry point scripts
├── configs/             # YAML configurations
├── models/              # Trained model checkpoints
├── logs/                # Training logs
└── tests/               # Unit tests
```

## 11. References

- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [GRU Paper: Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078)
- [UAV Trajectory System Documentation](../document/GRU_无人机时序轨迹网络详细说明.md)
