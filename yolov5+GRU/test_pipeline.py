"""
诊断与级联测试脚本 (test_pipeline.py)
"""
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # 将当前路径加入搜索目录以支持同级 import
    current_dir = str(Path(__file__).parent.resolve())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    from pipeline import YoloGRUPipeline

    video_path = "F:/UAV Trajectory System/test  vedio/complicated background/6.mp4"
    gru_model = "F:/UAV Trajectory System/gru detect/models/gru_baseline.pth"
    output_video = os.path.join(current_dir, "outputs", "output_predicted_6.avi")
    
    print("\n==================================================")
    print("Starting YOLOv5 + GRU Cascade Pipeline Test Run...")
    print(f"Input Video: {video_path}")
    print(f"GRU Model  : {gru_model}")
    print(f"Output     : {output_video}")
    print("==================================================\n")
    
    # 建立整合管道，加载用户提供的最新 YOLOv5 模型权重进行推理
    pipeline = YoloGRUPipeline(
        video_path=video_path,
        yolo_model_path="F:/UAV Trajectory System/yolov5_model/best include small.pt",
        gru_model_path=gru_model,
        conf_thres=0.4,
        nms_thres=0.45,
        device="cpu" # 纯 CPU 验证即可
    )
    
    # 仅测试运行前 350 帧，足以完整复现和验证全部轨迹预测和绘制细节
    pipeline.run(output_video, max_frames=350)
    
    print("\n==================================================")
    print("Test run completed successfully!")
    print(f"Output file size: {os.path.getsize(output_video) / (1024*1024):.2f} MB")
    print("==================================================\n")

if __name__ == "__main__":
    main()
