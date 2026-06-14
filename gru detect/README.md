# UAV Trajectory Prediction GUI

This folder contains the desktop prediction tool for UAV trajectory visualization.

## Main Entry

Use this one-click launcher:

```text
启动UAV预测窗口.vbs
```

The launcher starts `predict_gui.py` with `pythonw` in the `yolo` conda environment, so no extra terminal window is shown.

## Folder Layout

```text
detect/
├── models/                  # GRU model weights
├── outputs/                 # Per-video prediction archives
├── src/
│   ├── model.py             # UAVTrajectoryNet model definition
│   ├── tracker.py           # centroid tracking and feature extraction
│   ├── inference.py         # frame-by-frame realtime prediction session
│   └── paths.py             # output archive naming helpers
├── predict_gui.py           # PyQt5 real-time prediction GUI
├── requirements.txt
├── README.md
└── 启动UAV预测窗口.vbs       # one-click hidden-console launcher
```

## Output Naming

Prediction output is saved under:

```text
outputs/<archive_name>/<archive_name>_predicted.avi
```

If the imported video is named `rgb.mp4`, `video.mp4`, `input.mp4`, or `source.mp4`, the archive name uses the parent folder name instead of the generic video filename.

Example:

```text
...\episode_diagonal_d400_medium_forest_edge_medium_seed0114\rgb.mp4
```

outputs to:

```text
outputs/episode_diagonal_d400_medium_forest_edge_medium_seed0114/
```

## CSV Behavior

`measured_tracks.csv` is optional and is used as measured trajectory input for alignment and feature extraction. If a matching `ideal_tracks.csv` exists in the same scenario folder, the GUI also displays ground truth and saves ADE difference details to:

```text
outputs/<archive_name>/prediction_metrics.csv
```

## Shutdown Behavior

The GUI processes video frame by frame with a Qt timer. If you click stop or close the window after preview has started, the current output writer is released immediately and the partial prediction video is validated and archived under `outputs/<archive_name>/`.

The launcher and GUI set `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` before loading OpenCV/PyTorch. This avoids the known Windows `libiomp5md.dll` duplicate-runtime crash caused by mixed numerical libraries in the current conda environment.

Unhandled GUI exceptions are written to:

```text
predict_gui_error.log
```
