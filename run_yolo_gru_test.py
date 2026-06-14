"""
YOLO+GRU 动态集成测试启动器 (run_yolo_gru_test.py)
"""
import os
import sys

# 设置环境变量，防 OpenMP 冲突崩溃
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 动态寻找受 Windows GBK 乱码影响的物理文件夹
candidates = [d for d in os.listdir('.') 
              if 'agent' not in d 
              and 'detect' not in d 
              and 'train' not in d 
              and 'export' not in d 
              and 'yolov5_model' not in d 
              and 'test' not in d 
              and 'document' not in d 
              and 'distance' not in d 
              and os.path.isdir(d)]

if not candidates:
    print("Error: Target integration directory not found.")
    sys.exit(1)

target_dir = candidates[0]
print(f"--> [Launcher] Target physical directory located: {target_dir}")

# 加入搜索路径并执行
sys.path.insert(0, target_dir)

import test_pipeline
test_pipeline.main()
