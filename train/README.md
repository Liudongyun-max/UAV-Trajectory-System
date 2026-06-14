# UAV Trajectory Training Module

This module provides a complete training pipeline for the UAV trajectory recognition model.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python scripts/train.py --config configs/gru_baseline.yaml

# Evaluate the model
python scripts/evaluate.py --model models/gru_baseline_final.pth --data_dir ../export/input --visualize

# Export to ONNX
python scripts/export_model.py --model models/gru_baseline_final.pth --format onnx
```

## Documentation

- [Training Guide](TRAINING_GUIDE.md) - Complete training documentation
- [Project Documentation](../PROJECT_DOCUMENTATION.md) - Full system documentation

## Module Structure

```
train/
├── training/           # Core training code
├── scripts/            # Entry point scripts
├── configs/            # Configuration files
├── models/             # Model checkpoints
├── logs/               # Training logs
└── tests/              # Unit tests
```
