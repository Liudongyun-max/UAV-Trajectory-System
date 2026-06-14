#!/usr/bin/env python3
import cv2
import threading
import queue
import time
import numpy as np
import subprocess
import math
from collections import deque
from yololib import YoloRKNN 
from comms import DataSender, VideoSender

try:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# ==========================================
# 1
# ==========================================
N_CAM = 5                        
INIT_TIME = 5                    

TRACKER_MIN_HITS = 4            
TRACKER_MAX_DIST = 100           
TRACKER_MAX_GATE = 320           
TRACKER_RECENT_WINDOW = 18       
TRACKER_MIN_RECENT_HITS = 3      
TRACKER_CONFIRM_SCORE = 0.50     
TRACKER_MIN_TRAJ_SCORE = 0.55
TRACKER_MATCH_SCORE = 0.25
TRACKER_TEMPLATE_SIZE = 31       
TRACKER_MAX_SPEED_MPS = 10.0     
CONF_THRESH = 0.45               
YOLO_DIRECT_CONFIRM_HITS = 3
YOLO_DIRECT_CONFIRM_RECENT_HITS = 3
YOLO_DIRECT_CONFIRM_SCORE = 0.40
YOLO_DIRECT_CONFIRM_MAX_MISSES = 0
YOLO_TRACK_MAX_CONFIRMED_MISSES = 2
YOLO_TRACK_MAX_SEARCH_MISSES = 8
TRACK_SEARCH_PREDICT_MAX_MISSES = 2
MISS_VELOCITY_DECAY = 0.55
MAX_TRACK_ROIS_PER_FRAME = 3
SHOW_INDIVIDUAL_WINDOWS = True
VIDEO_SEND_EVERY_N_FRAMES = 3
MAX_DRAW_BOXES = 80
CAPTURE_W, CAPTURE_H = 2560, 1440

BOARD_ID = "BOARD_2" 
BOARD_ROW_IDX = 1 

DATA_TARGETS = {
    "gimbal": ("192.168.0.100", 8888)
}
VIDEO_TARGET_IP = "192.168.0.200"       
VIDEO_BASE_PORT = 9999                  

CROP_SIZE = 640      
LOW_LAYER_ROW_IDX = 0
HIGH_LAYER_MODE = BOARD_ROW_IDX != LOW_LAYER_ROW_IDX
ENABLE_TRAJECTORY_TRACKING = HIGH_LAYER_MODE
TRACK_SEARCH_MIN_YOLO_HITS = 2 if HIGH_LAYER_MODE else 3
TRACK_SEARCH_MIN_RECENT_HITS = 2 if HIGH_LAYER_MODE else 3
TRACK_SEARCH_MIN_SCORE = 0.32 if HIGH_LAYER_MODE else 0.40
TRACK_SEARCH_CONFIRMED_ONLY = True
ENABLE_LOW_LAYER_STATIC_BG_MASK = True
ENABLE_TIGHT_MOTION_ROI = not HIGH_LAYER_MODE
TIGHT_MOTION_ROI_PAD = 32
TIGHT_MOTION_ROI_FILL = 114
PROCESS_EVERY_N_FRAMES = 3
LOW_BG_SAMPLE_FRAMES = 60
LOW_BG_ABS_DELTA = 14
LOW_BG_STD_MULT = 3.0
DIFF_THRESH = 6 if HIGH_LAYER_MODE else 8
MIN_LOCAL_DIFF_MEAN = 11.0 if HIGH_LAYER_MODE else 16.0
ENABLE_GRAY_NOISE_SUPPRESSOR = not HIGH_LAYER_MODE
ENABLE_FUSION_INFERENCE = True
ENABLE_FUSION_DISPLAY = True
FUSION_REQUIRE_TRACK_MOTION = True
FUSION_TRACK_MIN_PIXELS = 3
FUSION_TRACK_CENTER_SIZE = 120
ENABLE_VERTICAL_STRIP_FILTER = True
VERTICAL_STRIP_ASPECT_RATIO = 1.6
VERTICAL_STRIP_MIN_HEIGHT = 8
GRAY_TEXTURE_PAD = 28
MAX_LOCAL_GRAY_STD = 38.0 if HIGH_LAYER_MODE else 30.0
BRIGHT_SPOT_ABS_THRESH = 210
BRIGHT_SPOT_REL_THRESH = 32.0
BRIGHT_SPOT_MAX_AREA = 220 if HIGH_LAYER_MODE else 140
BRIGHT_SPOT_MIN_BG_STD = 10.0
ENABLE_LCM_FILTER = True
LCM_BG_PAD = 18
LCM_MIN_SCORE = 1.8 if HIGH_LAYER_MODE else 2.4
LCM_MIN_RATIO = 1.25 if HIGH_LAYER_MODE else 1.45
LCM_REQUIRE_BOTH = not HIGH_LAYER_MODE
LCM_SCORE_WEIGHT = 3.0 if HIGH_LAYER_MODE else 2.0
MIN_OBJ_SIZE = 20
MOTION_ERODE_ITER = 1
MOTION_DILATE_ITER = 1
MOTION_CLOSE_ITER = 1
MOTION_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
MOTION_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
MOTION_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MAX_ROIS_PER_FRAME = 6 if HIGH_LAYER_MODE else 4
ENABLE_ROI_GRID_QUOTA = True
ROI_GRID_COLS = 4
ROI_GRID_ROWS = 3
ROI_GRID_MAX_PER_CELL = 1
INF_QUEUE_SIZE = MAX_ROIS_PER_FRAME + MAX_TRACK_ROIS_PER_FRAME + 2
RES_QUEUE_SIZE = MAX_ROIS_PER_FRAME + 4
DIFF_W, DIFF_H = 1920, 1080
MODEL_PATH = '../model/yolov5s.rknn'
ENABLE_GLOBAL_MOTION_COMP = False
ENABLE_EDGE_DENSITY_FILTER = False
STAB_W, STAB_H = 480, 270
STAB_MIN_RESPONSE = 0.08
STAB_MAX_SHIFT = 80
MAX_DIFF_AREA = 1000 if HIGH_LAYER_MODE else 600
MAX_DIFF_BOX_W = 180 if HIGH_LAYER_MODE else 120
MAX_DIFF_BOX_H = 180 if HIGH_LAYER_MODE else 120
NEAR_MAX_DIFF_AREA = 10000 if HIGH_LAYER_MODE else 6000
NEAR_MAX_DIFF_BOX_W = 450 if HIGH_LAYER_MODE else 320
NEAR_MAX_DIFF_BOX_H = 450 if HIGH_LAYER_MODE else 320
NEAR_MIN_COMPACTNESS = 0.04 if HIGH_LAYER_MODE else 0.06
NEAR_EDGE_DENSITY_THRESH = 0.45 if HIGH_LAYER_MODE else 0.30
EDGE_GRAD_THRESH = 45
EDGE_DENSITY_THRESH = 0.35 if HIGH_LAYER_MODE else 0.18
EDGE_DENSITY_PAD = 24

ROUGH_TARGET_WIDTH_M = 0.5
CAM_H_FOV = 17.5
IMG_W, IMG_H = CAPTURE_W, CAPTURE_H
DIFF_SCALE_AREA = (DIFF_W * DIFF_H) / float(IMG_W * IMG_H)
HIGH_LAYER_MIN_DIFF_AREA = max(2, int(MIN_OBJ_SIZE * DIFF_SCALE_AREA))
LOW_LAYER_MIN_DIFF_AREA = 50
MIN_DIFF_AREA = HIGH_LAYER_MIN_DIFF_AREA if HIGH_LAYER_MODE else LOW_LAYER_MIN_DIFF_AREA
FAR_MIN_COMPACTNESS = 0.0 if HIGH_LAYER_MODE else 0.18

ROUGH_RANGE_MIN_M = 20
ROUGH_RANGE_MAX_M = 2000
ROUGH_RANGE_ROUND_M = 10
CAM_MAP = {i: f"00000000{i+1}" for i in range(5)}

# ==========================================
# 2
# ==========================================
data_sender = DataSender(DATA_TARGETS, BOARD_ID)
video_senders =[VideoSender(VIDEO_TARGET_IP, VIDEO_BASE_PORT) for i in range(N_CAM)]

inf_queues = [queue.Queue(maxsize=INF_QUEUE_SIZE) for _ in range(N_CAM)]
res_queues =[queue.Queue(maxsize=RES_QUEUE_SIZE) for _ in range(N_CAM)]
display_queues =[queue.Queue(maxsize=2) for _ in range(N_CAM)]

stop_event = threading.Event()
SYSTEM_START_TIME = time.time()
INIT_SIGNAL_SENT = False
VIDEO_STREAM_ALLOWED = False

def put_latest(q, item):
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            return False

# ==========================================
# 3.
# ==========================================
def estimate_rough_range_m(box, frame_width=IMG_W):
    pixel_width = max(1.0, float(box[2] - box[0]))
    frame_width = max(1.0, float(frame_width))
    fx_px = (frame_width / 2.0) / math.tan(math.radians(CAM_H_FOV / 2.0))
    distance = (ROUGH_TARGET_WIDTH_M * fx_px) / pixel_width
    distance = max(ROUGH_RANGE_MIN_M, min(ROUGH_RANGE_MAX_M, distance))
    return int(round(distance / ROUGH_RANGE_ROUND_M) * ROUGH_RANGE_ROUND_M)

def merge_nearby_boxes(boxes, dist_thresh=120):
    if not boxes: return []
    merged = []
    for box in boxes:
        is_dup = False
        bcx, bcy = (box[0]+box[2])/2, (box[1]+box[3])/2
        for idx, m_box in enumerate(merged):
            mcx, mcy = (m_box[0]+m_box[2])/2, (m_box[1]+m_box[3])/2
            dist = math.sqrt((bcx-mcx)**2 + (bcy-mcy)**2)
            if dist < dist_thresh:
                box_score = float(box[4]) if len(box) > 4 else 1.0
                merged_score = float(m_box[4]) if len(m_box) > 4 else 1.0
                if box_score > merged_score:
                    merged[idx] = box
                is_dup = True; break
        if not is_dup: merged.append(box)
    return merged

def build_static_bg_model(samples):
    if not samples:
        return None, None
    stack = np.stack(samples, axis=0).astype(np.float32)
    bg = np.median(stack, axis=0).astype(np.uint8)
    std = np.std(stack, axis=0)
    tol = np.clip(LOW_BG_ABS_DELTA + LOW_BG_STD_MULT * std, LOW_BG_ABS_DELTA, 255).astype(np.uint8)
    return bg, tol

def apply_static_bg_mask(mask, gray, bg, tol):
    if not ENABLE_LOW_LAYER_STATIC_BG_MASK or bg is None or tol is None:
        return mask
    bg_diff = cv2.absdiff(gray, bg)
    changed = (bg_diff > tol).astype(np.uint8) * 255
    return cv2.bitwise_and(mask, changed)

def cleanup_motion_mask(mask):
    if MOTION_ERODE_ITER > 0:
        mask = cv2.erode(mask, MOTION_ERODE_KERNEL, iterations=MOTION_ERODE_ITER)
    if MOTION_DILATE_ITER > 0:
        mask = cv2.dilate(mask, MOTION_DILATE_KERNEL, iterations=MOTION_DILATE_ITER)
    if MOTION_CLOSE_ITER > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MOTION_CLOSE_KERNEL, iterations=MOTION_CLOSE_ITER)
    return mask

def make_fused_roi(frame, mask_small, roi_box, full_w, full_h):
    x1, y1, x2, y2 = [int(v) for v in roi_box[:4]]
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return roi
    gray_roi = frame_to_gray(roi)
    if mask_small is None:
        mask_roi = np.zeros_like(gray_roi)
    else:
        sx = mask_small.shape[1] / float(max(1, full_w))
        sy = mask_small.shape[0] / float(max(1, full_h))
        mx1 = max(0, min(mask_small.shape[1] - 1, int(round(x1 * sx))))
        my1 = max(0, min(mask_small.shape[0] - 1, int(round(y1 * sy))))
        mx2 = max(0, min(mask_small.shape[1], int(round(x2 * sx))))
        my2 = max(0, min(mask_small.shape[0], int(round(y2 * sy))))
        mask_patch = mask_small[my1:my2, mx1:mx2]
        if mask_patch.size == 0:
            mask_roi = np.zeros_like(gray_roi)
        else:
            mask_roi = cv2.resize(mask_patch, (gray_roi.shape[1], gray_roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.merge([gray_roi, gray_roi, mask_roi])

def make_padded_roi_canvas(frame, mask_small, roi_info, full_w, full_h, fusion=True, crop_size=CROP_SIZE, fill_value=114):
    x1, y1, x2, y2 = [int(v) for v in roi_info[:4]]
    pad_x = int(roi_info[5]) if len(roi_info) > 5 else 0
    pad_y = int(roi_info[6]) if len(roi_info) > 6 else 0
    src_w = max(0, x2 - x1)
    src_h = max(0, y2 - y1)
    if src_w <= 0 or src_h <= 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    if fusion:
        patch = make_fused_roi(frame, mask_small, (x1, y1, x2, y2), full_w, full_h)
        canvas = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        canvas[:, :, 0] = np.uint8(fill_value)
        canvas[:, :, 1] = np.uint8(fill_value)
    else:
        patch = frame[y1:y2, x1:x2]
        canvas = np.full((crop_size, crop_size, 3), np.uint8(fill_value), dtype=np.uint8)

    if patch.size == 0:
        return patch

    dst_x1 = max(0, min(crop_size, pad_x))
    dst_y1 = max(0, min(crop_size, pad_y))
    dst_x2 = min(crop_size, dst_x1 + patch.shape[1])
    dst_y2 = min(crop_size, dst_y1 + patch.shape[0])
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return canvas
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = patch[:dst_y2 - dst_y1, :dst_x2 - dst_x1]
    return canvas

def make_fused_display_frame(frame, mask_small, full_w, full_h):
    gray = frame_to_gray(frame)
    if mask_small is None:
        mask_full = np.zeros_like(gray)
    elif mask_small.shape[1] == gray.shape[1] and mask_small.shape[0] == gray.shape[0]:
        mask_full = mask_small
    else:
        mask_full = cv2.resize(mask_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.merge([gray, gray, mask_full])

def track_roi_has_motion(track_roi, mask_small, full_w, full_h, min_pixels=FUSION_TRACK_MIN_PIXELS, center_size=FUSION_TRACK_CENTER_SIZE):
    if mask_small is None:
        return True
    x1, y1, x2, y2 = [float(v) for v in track_roi[:4]]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    sx = mask_small.shape[1] / float(max(1, full_w))
    sy = mask_small.shape[0] / float(max(1, full_h))
    half = float(center_size) * 0.5
    mx1 = int(max(0, round((cx - half) * sx)))
    my1 = int(max(0, round((cy - half) * sy)))
    mx2 = int(min(mask_small.shape[1], round((cx + half) * sx)))
    my2 = int(min(mask_small.shape[0], round((cy + half) * sy)))
    if mx2 <= mx1 or my2 <= my1:
        return False
    return cv2.countNonZero(mask_small[my1:my2, mx1:mx2]) >= int(min_pixels)

def limit_mask_to_rois(mask_small, rois, full_w, full_h):
    if mask_small is None or not rois:
        return None
    limited = np.zeros_like(mask_small)
    sx = mask_small.shape[1] / float(max(1, full_w))
    sy = mask_small.shape[0] / float(max(1, full_h))
    for roi in rois:
        x1, y1, x2, y2 = [float(v) for v in roi[:4]]
        mx1 = int(max(0, min(mask_small.shape[1], math.floor(x1 * sx))))
        my1 = int(max(0, min(mask_small.shape[0], math.floor(y1 * sy))))
        mx2 = int(max(0, min(mask_small.shape[1], math.ceil(x2 * sx))))
        my2 = int(max(0, min(mask_small.shape[0], math.ceil(y2 * sy))))
        if mx2 <= mx1 or my2 <= my1:
            continue
        limited[my1:my2, mx1:mx2] = mask_small[my1:my2, mx1:mx2]
    return limited

def boost_roi_score(roi, boost):
    values = list(roi)
    if len(values) < 5:
        values.append(0.0)
    values[4] = float(values[4]) + float(boost)
    return tuple(values)

def crop_roi_from_center(cx, cy, full_w, full_h, crop_size=CROP_SIZE):
    half = crop_size // 2
    x1 = max(0, int(round(cx)) - half)
    y1 = max(0, int(round(cy)) - half)
    x2 = min(full_w, x1 + crop_size)
    y2 = min(full_h, y1 + crop_size)
    x1 = max(0, x2 - crop_size)
    y1 = max(0, y2 - crop_size)
    return [x1, y1, x2, y2]

def frame_to_gray(frame):
    if frame is None:
        return None
    if len(frame.shape) == 2:
        return frame
    if len(frame.shape) == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def align_previous_gray(prev_gray, curr_gray):
    if not ENABLE_GLOBAL_MOTION_COMP or prev_gray is None:
        return prev_gray
    try:
        prev_small = cv2.resize(prev_gray, (STAB_W, STAB_H))
        curr_small = cv2.resize(curr_gray, (STAB_W, STAB_H))
        shift, response = cv2.phaseCorrelate(np.float32(prev_small), np.float32(curr_small))
        dx = float(shift[0]) * (prev_gray.shape[1] / float(STAB_W))
        dy = float(shift[1]) * (prev_gray.shape[0] / float(STAB_H))
        if response < STAB_MIN_RESPONSE or abs(dx) > STAB_MAX_SHIFT or abs(dy) > STAB_MAX_SHIFT:
            return prev_gray
        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(prev_gray, mat, (prev_gray.shape[1], prev_gray.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return prev_gray

def build_edge_mask(gray):
    if not ENABLE_EDGE_DENSITY_FILTER:
        return None
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)
    edge_mask = (grad >= EDGE_GRAD_THRESH).astype(np.uint8) * 255
    return cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)

def edge_density(edge_mask, x, y, w, h):
    if edge_mask is None:
        return 0.0
    x1 = max(0, x - EDGE_DENSITY_PAD)
    y1 = max(0, y - EDGE_DENSITY_PAD)
    x2 = min(edge_mask.shape[1], x + w + EDGE_DENSITY_PAD)
    y2 = min(edge_mask.shape[0], y + h + EDGE_DENSITY_PAD)
    patch = edge_mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(cv2.countNonZero(patch)) / float(patch.size)

def local_contrast_measure(diff_img, mask_patch, x, y, w, h):
    if not ENABLE_LCM_FILTER or diff_img is None or mask_patch is None:
        return 999.0, 999.0
    pad = max(int(LCM_BG_PAD), int(max(w, h) * 2))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(diff_img.shape[1], x + w + pad)
    y2 = min(diff_img.shape[0], y + h + pad)
    local_patch = diff_img[y1:y2, x1:x2].astype(np.float32)
    if local_patch.size == 0:
        return 999.0, 999.0

    active = mask_patch > 0
    target_vals = diff_img[y:y+h, x:x+w][active].astype(np.float32) if np.any(active) else diff_img[y:y+h, x:x+w].astype(np.float32).reshape(-1)
    if target_vals.size == 0:
        return 0.0, 0.0

    ring_mask = np.ones(local_patch.shape, dtype=np.uint8)
    rx1 = max(0, x - x1)
    ry1 = max(0, y - y1)
    rx2 = min(ring_mask.shape[1], rx1 + w)
    ry2 = min(ring_mask.shape[0], ry1 + h)
    ring_mask[ry1:ry2, rx1:rx2] = 0
    ring_vals = local_patch[ring_mask > 0]
    if ring_vals.size == 0:
        return 999.0, 999.0

    target_mean = float(np.mean(target_vals))
    bg_mean = float(np.mean(ring_vals))
    bg_std = float(np.std(ring_vals))
    lcm_score = (target_mean - bg_mean) / max(bg_std, 1.0)
    lcm_ratio = target_mean / max(bg_mean, 1.0)
    return lcm_score, lcm_ratio

def is_vertical_strip_box(box_or_wh):
    if not ENABLE_VERTICAL_STRIP_FILTER:
        return False
    if len(box_or_wh) >= 4:
        bw = float(box_or_wh[2] - box_or_wh[0])
        bh = float(box_or_wh[3] - box_or_wh[1])
    else:
        bw, bh = float(box_or_wh[0]), float(box_or_wh[1])
    if bw <= 0 or bh <= 0:
        return False
    return bh >= VERTICAL_STRIP_MIN_HEIGHT and bh >= bw * VERTICAL_STRIP_ASPECT_RATIO

def prioritize_diverse_rois(rois, full_w, full_h, max_rois=MAX_ROIS_PER_FRAME):
    if not ENABLE_ROI_GRID_QUOTA or not rois:
        return rois
    sorted_rois = sorted(rois, key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
    selected, overflow = [], []
    cell_counts = {}
    cols = max(1, int(ROI_GRID_COLS))
    rows = max(1, int(ROI_GRID_ROWS))
    max_per_cell = max(1, int(ROI_GRID_MAX_PER_CELL))
    for roi in sorted_rois:
        cx = (float(roi[0]) + float(roi[2])) * 0.5
        cy = (float(roi[1]) + float(roi[3])) * 0.5
        cell_x = max(0, min(cols - 1, int(cx / max(1.0, float(full_w)) * cols)))
        cell_y = max(0, min(rows - 1, int(cy / max(1.0, float(full_h)) * rows)))
        cell = (cell_x, cell_y)
        if cell_counts.get(cell, 0) < max_per_cell:
            selected.append(roi)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
        else:
            overflow.append(roi)
    return (selected[:max_rois] + overflow) if len(selected) >= max_rois else (selected + overflow)

def should_suppress_gray_noise(gray, mask_patch, x, y, w, h, area):
    if not ENABLE_GRAY_NOISE_SUPPRESSOR or gray is None:
        return False
    x1 = max(0, x - GRAY_TEXTURE_PAD)
    y1 = max(0, y - GRAY_TEXTURE_PAD)
    x2 = min(gray.shape[1], x + w + GRAY_TEXTURE_PAD)
    y2 = min(gray.shape[0], y + h + GRAY_TEXTURE_PAD)
    local_patch = gray[y1:y2, x1:x2]
    if local_patch.size == 0:
        return False

    local_std = float(np.std(local_patch))
    if local_std > MAX_LOCAL_GRAY_STD:
        return True

    active = mask_patch > 0
    if not np.any(active):
        return False
    active_vals = gray[y:y+h, x:x+w][active]
    if active_vals.size == 0:
        return False

    local_median = float(np.median(local_patch))
    bright_thresh = max(float(BRIGHT_SPOT_ABS_THRESH), local_median + BRIGHT_SPOT_REL_THRESH)
    active_mean = float(np.mean(active_vals))
    bright_ratio = float(np.count_nonzero(active_vals >= bright_thresh)) / float(active_vals.size)
    return (
        area <= BRIGHT_SPOT_MAX_AREA
        and bright_ratio >= 0.55
        and active_mean >= local_median + BRIGHT_SPOT_REL_THRESH
        and (local_std >= BRIGHT_SPOT_MIN_BG_STD or bright_ratio >= 0.80)
    )

def motion_rois_from_mask(mask, diff_img, edge_mask, gray, scale_x, scale_y, full_w, full_h):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_rois = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        if is_vertical_strip_box((w, h)):
            continue
        mask_patch = mask[y:y+h, x:x+w]
        area = int(cv2.countNonZero(mask_patch))
        if area < MIN_DIFF_AREA:
            continue
        texture = edge_density(edge_mask, x, y, w, h)
        active = mask_patch > 0
        local_diff = float(diff_img[y:y+h, x:x+w][active].mean()) if np.any(active) else 0.0
        if local_diff < MIN_LOCAL_DIFF_MEAN:
            continue
        lcm_score, lcm_ratio = local_contrast_measure(diff_img, mask_patch, x, y, w, h)
        if ENABLE_LCM_FILTER:
            if LCM_REQUIRE_BOTH:
                if lcm_score < LCM_MIN_SCORE or lcm_ratio < LCM_MIN_RATIO:
                    continue
            elif lcm_score < LCM_MIN_SCORE and lcm_ratio < LCM_MIN_RATIO:
                continue
        if should_suppress_gray_noise(gray, mask_patch, x, y, w, h, area):
            continue
        compactness = area / max(1.0, float(w * h))

        is_far_tiny = (
            area <= MAX_DIFF_AREA
            and w <= MAX_DIFF_BOX_W
            and h <= MAX_DIFF_BOX_H
            and compactness >= FAR_MIN_COMPACTNESS
            and texture <= EDGE_DENSITY_THRESH
        )
        is_near_compact = (
            area <= NEAR_MAX_DIFF_AREA
            and w <= NEAR_MAX_DIFF_BOX_W
            and h <= NEAR_MAX_DIFF_BOX_H
            and compactness >= NEAR_MIN_COMPACTNESS
            and texture <= NEAR_EDGE_DENSITY_THRESH
        )
        if not (is_far_tiny or is_near_compact):
            continue

        if is_near_compact and not is_far_tiny:
            score = local_diff + 18.0 * compactness - 0.003 * area - 18.0 * texture
        else:
            score = local_diff + 10.0 * compactness - 0.015 * area - 20.0 * texture
        score += LCM_SCORE_WEIGHT * max(0.0, min(float(lcm_score), 8.0))

        fx, fy = int(math.floor(x * scale_x)), int(math.floor(y * scale_y))
        fx2 = int(math.ceil((x + w) * scale_x))
        fy2 = int(math.ceil((y + h) * scale_y))
        fw, fh = max(1, fx2 - fx), max(1, fy2 - fy)
        cx, cy = fx + fw // 2, fy + fh // 2
        if ENABLE_TIGHT_MOTION_ROI:
            pad = max(0, int(TIGHT_MOTION_ROI_PAD))
            sx1 = max(0, fx - pad)
            sy1 = max(0, fy - pad)
            sx2 = min(full_w, fx2 + pad)
            sy2 = min(full_h, fy2 + pad)
            sw, sh = sx2 - sx1, sy2 - sy1
            if sw <= CROP_SIZE and sh <= CROP_SIZE:
                px = (CROP_SIZE - sw) // 2
                py = (CROP_SIZE - sh) // 2
                temp_rois.append((
                    sx1,
                    sy1,
                    sx2,
                    sy2,
                    score,
                    px,
                    py,
                ))
                continue
        rx1, ry1, rx2, ry2 = crop_roi_from_center(cx, cy, full_w, full_h)
        temp_rois.append((
            rx1,
            ry1,
            rx2,
            ry2,
            score,
        ))
    temp_rois.sort(key=lambda r: r[4], reverse=True)
    rois = merge_nearby_boxes(temp_rois, dist_thresh=300)
    rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
    return prioritize_diverse_rois(rois, full_w, full_h)

class TrajectoryFilter:
    def __init__(
        self,
        max_dist=100,
        min_hits=3,
        recent_window=15,
        min_recent_hits=2,
        confirm_score=0.45,
        template_size=31,
        max_gate=320,
        max_speed_mps=10.0,
        fps=15.0,
    ):
        self.trackers = []
        self.next_id = 1
        self.max_dist = float(max_dist)
        self.min_hits = int(min_hits)
        self.recent_window = int(recent_window)
        self.min_recent_hits = int(min_recent_hits)
        self.confirm_score = float(confirm_score)
        self.template_size = int(template_size if template_size % 2 == 1 else template_size + 1)
        self.max_gate = float(max_gate)
        self.max_speed_mps = float(max_speed_mps)
        self.fps = max(1.0, float(fps))

    def _center(self, box):
        return (float(box[0] + box[2]) * 0.5, float(box[1] + box[3]) * 0.5)

    def _box_wh(self, box):
        return (max(1.0, float(box[2] - box[0])), max(1.0, float(box[3] - box[1])))

    def _det_score(self, box):
        return float(box[4]) if len(box) > 4 else CONF_THRESH

    def _clamp01(self, value):
        return max(0.0, min(1.0, float(value)))

    def _crop_template(self, frame, box):
        if frame is None: return None
        h, w = frame.shape[:2]
        cx, cy = self._center(box)
        r = self.template_size // 2
        x1 = max(0, int(round(cx)) - r)
        y1 = max(0, int(round(cy)) - r)
        x2 = min(w, int(round(cx)) + r + 1)
        y2 = min(h, int(round(cy)) + r + 1)
        patch = frame[y1:y2, x1:x2]
        if patch.shape[0] < 5 or patch.shape[1] < 5:
            return None
        gray = frame_to_gray(patch)
        gray = cv2.resize(gray, (self.template_size, self.template_size))
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def _template_score(self, trk, frame, det):
        if trk.get('template') is None:
            return 0.65
        patch = self._crop_template(frame, det)
        if patch is None:
            return 0.55
        if float(np.std(patch)) < 1.0 or float(np.std(trk['template'])) < 1.0:
            return 0.55
        raw = cv2.matchTemplate(patch, trk['template'], cv2.TM_CCOEFF_NORMED)[0][0]
        if not np.isfinite(raw):
            return 0.55
        return self._clamp01((float(raw) + 1.0) * 0.5)

    def _predict_center(self, trk, frame_idx):
        dt = max(1, int(frame_idx) - int(trk['last_frame']))
        return trk['cx'] + trk['vx'] * dt, trk['cy'] + trk['vy'] * dt, dt

    def _predicted_box(self, trk, frame_idx):
        cx, cy, _ = self._predict_center(trk, frame_idx)
        w, h = trk['w'], trk['h']
        return [int(round(cx - w * 0.5)), int(round(cy - h * 0.5)), int(round(cx + w * 0.5)), int(round(cy + h * 0.5))]

    def _dynamic_gate(self, trk, frame_idx):
        _, _, dt = self._predict_center(trk, frame_idx)
        speed_px_frame = math.sqrt(trk['vx'] * trk['vx'] + trk['vy'] * trk['vy'])
        width_px = max(1.0, trk['w'])
        physical_motion_px = self.max_speed_mps * (dt / self.fps) * width_px / max(ROUGH_TARGET_WIDTH_M, 0.01)
        gate = self.max_dist + 0.35 * speed_px_frame * dt + 0.75 * physical_motion_px + 12.0 * trk['misses']
        return max(self.max_dist, min(self.max_gate, gate))

    def _motion_score(self, trk, det, frame_idx):
        cx, cy = self._center(det)
        dt = max(1, int(frame_idx) - int(trk['last_frame']))
        nvx = (cx - trk['cx']) / dt
        nvy = (cy - trk['cy']) / dt
        old_speed = math.sqrt(trk['vx'] * trk['vx'] + trk['vy'] * trk['vy'])
        new_speed = math.sqrt(nvx * nvx + nvy * nvy)
        if old_speed < 2.0 or new_speed < 2.0:
            return 0.75
        speed_change = abs(new_speed - old_speed) / max(old_speed, 8.0)
        dot = trk['vx'] * nvx + trk['vy'] * nvy
        cos_v = dot / max(old_speed * new_speed, 1e-6)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_v))))
        return self._clamp01(1.0 - 0.35 * min(1.0, speed_change) - 0.45 * min(1.0, angle / 150.0))

    def _trajectory_score(self, trk):
        pts = list(trk['history'])
        if len(pts) < 4:
            return 0.7
        steps = [math.sqrt((pts[i][0] - pts[i-1][0])**2 + (pts[i][1] - pts[i-1][1])**2) for i in range(1, len(pts))]
        if not steps:
            return 0.7
        median_step = float(np.median(steps))
        max_step = max(steps)
        score = 1.0
        if median_step > 1.0 and max_step > median_step * 3.0 + 35.0:
            score -= 0.35
        headings = []
        for i in range(1, len(pts)):
            dx, dy = pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]
            if abs(dx) + abs(dy) > 1.0:
                headings.append(math.degrees(math.atan2(dy, dx)))
        if len(headings) >= 4:
            turns = [abs((headings[i] - headings[i-1] + 180.0) % 360.0 - 180.0) for i in range(1, len(headings))]
            sharp_turns = sum(1 for t in turns if t > 130.0)
            if sharp_turns >= 2:
                score -= 0.25
        recent_hits = sum(trk['hit_history'])
        if len(trk['hit_history']) >= self.recent_window and recent_hits < self.min_recent_hits:
            score -= 0.35
        return self._clamp01(score)

    def _match_score(self, trk, det, frame, frame_idx):
        pred_cx, pred_cy, _ = self._predict_center(trk, frame_idx)
        cx, cy = self._center(det)
        dist = math.sqrt((cx - pred_cx)**2 + (cy - pred_cy)**2)
        gate = self._dynamic_gate(trk, frame_idx)
        if dist > gate:
            return None
        distance_score = self._clamp01(1.0 - dist / max(gate, 1.0))
        yolo_score = self._clamp01(self._det_score(det))
        tmpl_score = self._template_score(trk, frame, det)
        motion_score = self._motion_score(trk, det, frame_idx)
        score = 0.42 * distance_score + 0.23 * yolo_score + 0.17 * tmpl_score + 0.18 * motion_score
        return score, dist, tmpl_score

    def _start_track(self, det, frame, frame_idx):
        cx, cy = self._center(det)
        w, h = self._box_wh(det)
        trk = {
            'id': self.next_id,
            'box': det[:4],
            'cx': cx,
            'cy': cy,
            'w': w,
            'h': h,
            'vx': 0.0,
            'vy': 0.0,
            'hits': 1,
            'misses': 0,
            'last_frame': int(frame_idx),
            'score': 0.35 + 0.35 * self._clamp01(self._det_score(det)),
            'yolo_hits': 1,
            'history': deque([(cx, cy)], maxlen=self.recent_window),
            'hit_history': deque([1], maxlen=self.recent_window),
            'template': self._crop_template(frame, det),
        }
        self.next_id += 1
        self.trackers.append(trk)

    def _update_track(self, trk, det, frame, frame_idx, match_score, tmpl_score):
        cx, cy = self._center(det)
        dt = max(1, int(frame_idx) - int(trk['last_frame']))
        nvx = (cx - trk['cx']) / dt
        nvy = (cy - trk['cy']) / dt
        alpha = 0.55
        trk['vx'] = alpha * nvx + (1.0 - alpha) * trk['vx']
        trk['vy'] = alpha * nvy + (1.0 - alpha) * trk['vy']
        trk['cx'], trk['cy'] = cx, cy
        trk['w'], trk['h'] = self._box_wh(det)
        trk['box'] = det[:4]
        trk['last_frame'] = int(frame_idx)
        trk['hits'] += 1
        trk['yolo_hits'] = trk.get('yolo_hits', 0) + 1
        trk['misses'] = 0
        trk['score'] = self._clamp01(0.75 * trk['score'] + 0.25 * match_score)
        trk['history'].append((cx, cy))
        trk['hit_history'].append(1)
        new_template = self._crop_template(frame, det)
        if new_template is not None:
            if trk.get('template') is None:
                trk['template'] = new_template
            elif tmpl_score >= 0.45 and match_score >= 0.45:
                trk['template'] = cv2.addWeighted(trk['template'], 0.85, new_template, 0.15, 0)

    def _miss_track(self, trk):
        trk['misses'] += 1
        trk['vx'] *= MISS_VELOCITY_DECAY
        trk['vy'] *= MISS_VELOCITY_DECAY
        trk['score'] = self._clamp01(trk['score'] * 0.92)
        trk['hit_history'].append(0)

    def _is_confirmed(self, trk):
        recent_hits = sum(trk['hit_history'])
        traj_score = self._trajectory_score(trk)
        if trk.get('yolo_hits', 0) >= YOLO_DIRECT_CONFIRM_HITS:
            return (
                recent_hits >= YOLO_DIRECT_CONFIRM_RECENT_HITS
                and trk['misses'] <= YOLO_DIRECT_CONFIRM_MAX_MISSES
                and trk['score'] >= YOLO_DIRECT_CONFIRM_SCORE
            )
        return (
            trk['hits'] >= self.min_hits
            and recent_hits >= self.min_recent_hits
            and trk['misses'] <= YOLO_TRACK_MAX_CONFIRMED_MISSES
            and trk['score'] >= self.confirm_score
            and traj_score >= TRACKER_MIN_TRAJ_SCORE
        )

    def update(self, detections, frame=None, frame_idx=0):
        detections = [list(d) for d in detections]
        pairs = []
        for ti, trk in enumerate(self.trackers):
            for di, det in enumerate(detections):
                result = self._match_score(trk, det, frame, frame_idx)
                if result is None:
                    continue
                score, dist, tmpl_score = result
                if score >= TRACKER_MATCH_SCORE:
                    pairs.append((score, dist, tmpl_score, ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_tracks, matched_dets = set(), set()
        for score, _, tmpl_score, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            self._update_track(self.trackers[ti], detections[di], frame, frame_idx, score, tmpl_score)
            matched_tracks.add(ti)
            matched_dets.add(di)

        for ti, trk in enumerate(self.trackers):
            if ti not in matched_tracks:
                self._miss_track(trk)

        for di, det in enumerate(detections):
            if di not in matched_dets:
                self._start_track(det, frame, frame_idx)

        kept = []
        for trk in self.trackers:
            if trk['misses'] <= YOLO_TRACK_MAX_SEARCH_MISSES and trk['score'] >= 0.12:
                kept.append(trk)
        self.trackers = kept
        return [t['box'] for t in self.get_confirmed_tracks(frame_idx)]

    def get_yolo_seeded_search_rois(self, frame_idx, full_w, full_h, max_rois=MAX_TRACK_ROIS_PER_FRAME):
        rois = []
        for trk in self.trackers:
            if trk['misses'] > YOLO_TRACK_MAX_SEARCH_MISSES:
                continue
            if TRACK_SEARCH_CONFIRMED_ONLY and not self._is_confirmed(trk):
                continue
            recent_hits = sum(trk.get('hit_history', []))
            if (
                trk.get('yolo_hits', 0) < TRACK_SEARCH_MIN_YOLO_HITS
                or recent_hits < TRACK_SEARCH_MIN_RECENT_HITS
                or float(trk.get('score', 0.0)) < TRACK_SEARCH_MIN_SCORE
            ):
                continue
            box = self._predicted_box(trk, frame_idx) if 0 < trk['misses'] <= TRACK_SEARCH_PREDICT_MAX_MISSES else trk['box']
            cx = (float(box[0]) + float(box[2])) * 0.5
            cy = (float(box[1]) + float(box[3])) * 0.5
            rx1, ry1, rx2, ry2 = crop_roi_from_center(cx, cy, full_w, full_h)
            priority = 1000.0 + float(trk.get('score', 0.0)) - 20.0 * float(trk['misses'])
            rois.append((rx1, ry1, rx2, ry2, priority))
        rois = merge_nearby_boxes(rois, dist_thresh=160)
        rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
        return rois[:max_rois]

    def get_confirmed_tracks(self, frame_idx=0):
        confirmed = []
        for trk in self.trackers:
            if not self._is_confirmed(trk):
                continue
            box = trk['box']
            history = list(trk['history'])
            confirmed.append({
                'id': trk['id'],
                'box': box,
                'history': history,
                'score': trk['score'],
                'live': trk['misses'] == 0,
                'misses': trk['misses'],
            })
        return confirmed

def get_camera_node(bus_info_keyword):
    try:
        out = subprocess.check_output("v4l2-ctl --list-devices", shell=True).decode("utf-8").split('\n')
        is_target = False
        for line in out:
            line = line.strip()
            if not line: continue
            if not line.startswith('/dev/video'): is_target = (bus_info_keyword in line)
            elif is_target and line.startswith('/dev/video'): return line
    except: pass
    return None

# ==========================================
# 4
# ==========================================
def inference_worker():
    try: 
        yolo = YoloRKNN(MODEL_PATH, (640, 640), CONF_THRESH, 0.45)
        print("--> RKNN Loaded.", flush=True)
    except Exception as e:
        print(f"[Fatal] RKNN Init Failed: {e}"); return
    
    while not stop_event.is_set():
        did_work = False
        for i in range(N_CAM):
            try:
                roi, x, y = inf_queues[i].get_nowait()
                did_work = True
                res = yolo.infer(roi)
                rh, rw = roi.shape[:2]
                if res is None:
                    res = []
                put_latest(res_queues[i], (res, x, y, rw, rh))
            except queue.Empty: pass
        if not did_work: time.sleep(0.001)
    if yolo: yolo.release()

# ==========================================
# 5
# ==========================================
def capture_job(cam_idx):
    if cam_idx not in CAM_MAP: return
    target_hw_id = CAM_MAP[cam_idx]
    dev_path = get_camera_node(target_hw_id) or f"/dev/video{cam_idx * 2}"

    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_t1 = None
    tracker = TrajectoryFilter(
        max_dist=TRACKER_MAX_DIST,
        min_hits=TRACKER_MIN_HITS,
        recent_window=TRACKER_RECENT_WINDOW,
        min_recent_hits=TRACKER_MIN_RECENT_HITS,
        confirm_score=TRACKER_CONFIRM_SCORE,
        template_size=TRACKER_TEMPLATE_SIZE,
        max_gate=TRACKER_MAX_GATE,
        max_speed_mps=TRACKER_MAX_SPEED_MPS,
        fps=15.0,
    ) 
    
    f_idx = 0
    v_sender = video_senders[cam_idx]
    learning_mode = True
    current_draw_boxes =[] 
    bg_samples = []
    bg_gray, bg_tol = None, None
    latest_fusion_mask = None
    last_tracker_update_frame = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: time.sleep(0.1); continue
        gray_frame = None
        if len(frame.shape) == 2:
            gray_frame = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            gray_frame = frame[:, :, 0]
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        H, W = frame.shape[:2]
        f_idx += 1
        visual_enabled = VIDEO_STREAM_ALLOWED or SHOW_INDIVIDUAL_WINDOWS
        send_video_this_frame = VIDEO_STREAM_ALLOWED and ((f_idx + cam_idx) % VIDEO_SEND_EVERY_N_FRAMES == 0)
        draw_this_frame = send_video_this_frame or SHOW_INDIVIDUAL_WINDOWS
        processed_for_detection = False
        sent_inference_this_frame = False
        if learning_mode and (time.time() - SYSTEM_START_TIME > INIT_TIME):
            learning_mode = False

        if (f_idx + cam_idx) % PROCESS_EVERY_N_FRAMES == 0:
            gray = gray_frame if gray_frame is not None else frame_to_gray(frame)
            if gray.shape[1] == DIFF_W and gray.shape[0] == DIFF_H:
                gray_small = gray
            else:
                gray_small = cv2.resize(gray, (DIFF_W, DIFF_H), interpolation=cv2.INTER_AREA)
            scale_x, scale_y = W / float(DIFF_W), H / float(DIFF_H)
            if ENABLE_LOW_LAYER_STATIC_BG_MASK and bg_gray is None:
                bg_samples.append(gray_small.copy())
                if len(bg_samples) >= LOW_BG_SAMPLE_FRAMES:
                    bg_gray, bg_tol = build_static_bg_model(bg_samples)
                    bg_samples = []
                    print(f"--> Cam {cam_idx} static background ready.", flush=True)
                frame_t1 = gray_small
                continue
            
            processed_for_detection = True
            rois =[]
            fusion_mask = None
            track_rois = tracker.get_yolo_seeded_search_rois(f_idx, W, H) if ENABLE_TRAJECTORY_TRACKING else []
            track_roi_keys = set()
            if frame_t1 is not None:
                aligned_t1 = align_previous_gray(frame_t1, gray_small)
                diff_img = cv2.absdiff(aligned_t1, gray_small)
                _, mask = cv2.threshold(diff_img, DIFF_THRESH, 255, cv2.THRESH_BINARY)
                mask = apply_static_bg_mask(mask, gray_small, bg_gray, bg_tol)
                mask = cleanup_motion_mask(mask)
                fusion_mask = mask
                latest_fusion_mask = fusion_mask
                edge_mask = build_edge_mask(gray_small)
                rois = motion_rois_from_mask(mask, diff_img, edge_mask, gray_small, scale_x, scale_y, W, H)
            if FUSION_REQUIRE_TRACK_MOTION and track_rois:
                track_rois = [r for r in track_rois if track_roi_has_motion(r, fusion_mask, W, H)]
            if track_rois:
                if ENABLE_TIGHT_MOTION_ROI and fusion_mask is not None:
                    track_mask = limit_mask_to_rois(fusion_mask, track_rois, W, H)
                    track_motion_rois = motion_rois_from_mask(track_mask, diff_img, edge_mask, gray_small, scale_x, scale_y, W, H) if track_mask is not None else []
                    track_motion_rois = [boost_roi_score(r, 1000.0) for r in track_motion_rois]
                    track_roi_keys = {tuple(int(v) for v in r[:4]) for r in track_motion_rois}
                    rois = merge_nearby_boxes(track_motion_rois + rois, dist_thresh=160)
                else:
                    track_roi_keys = {tuple(int(v) for v in r[:4]) for r in track_rois}
                    rois = merge_nearby_boxes(track_rois + rois, dist_thresh=160)
                rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
                rois = prioritize_diverse_rois(rois, W, H)
            
            frame_t1 = gray_small
            for roi in rois[:MAX_ROIS_PER_FRAME]:
                x1, y1, x2, y2 = roi[:4]
                use_padded_roi = ENABLE_TIGHT_MOTION_ROI and len(roi) > 6
                if use_padded_roi:
                    roi_img = make_padded_roi_canvas(
                        frame,
                        fusion_mask,
                        roi,
                        W,
                        H,
                        fusion=ENABLE_FUSION_INFERENCE,
                        fill_value=TIGHT_MOTION_ROI_FILL,
                    )
                    origin_x = int(x1) - int(roi[5])
                    origin_y = int(y1) - int(roi[6])
                else:
                    roi_img = make_fused_roi(frame, fusion_mask, (x1, y1, x2, y2), W, H) if ENABLE_FUSION_INFERENCE else frame[y1:y2, x1:x2].copy()
                    origin_x = int(x1)
                    origin_y = int(y1)
                if roi_img.size == 0:
                    continue
                if put_latest(inf_queues[cam_idx], (roi_img, origin_x, origin_y)):
                    sent_inference_this_frame = True
                    is_track_roi = tuple(int(v) for v in roi[:4]) in track_roi_keys
                    label = "TRKROI" if is_track_roi else "ROI"
                    color = (0, 255, 255) if is_track_roi else (255, 255, 255)
                    if visual_enabled:
                        current_draw_boxes.append([x1, y1, x2, y2, color, label, 2])


        raw_boxes_in_this_frame =[]
        got_inference_result = False
        while True:
            try:
                result = res_queues[cam_idx].get_nowait()
                got_inference_result = True
                if len(result) >= 5:
                    dets, xo, yo, roi_w, roi_h = result[:5]
                else:
                    dets, xo, yo = result
                    roi_w, roi_h = CROP_SIZE, CROP_SIZE
                sx = float(roi_w) / float(CROP_SIZE)
                sy = float(roi_h) / float(CROP_SIZE)
                for d in dets:
                    score = float(d[4]) if len(d) > 4 else CONF_THRESH
                    mapped_box = [
                        int(max(0, min(W, d[0] * sx + xo))),
                        int(max(0, min(H, d[1] * sy + yo))),
                        int(max(0, min(W, d[2] * sx + xo))),
                        int(max(0, min(H, d[3] * sy + yo))),
                        score,
                    ]
                    if (
                        mapped_box[2] > mapped_box[0]
                        and mapped_box[3] > mapped_box[1]
                        and not is_vertical_strip_box(mapped_box)
                    ):
                        raw_boxes_in_this_frame.append(mapped_box)
            except queue.Empty: break


        unique_raw_boxes = merge_nearby_boxes(raw_boxes_in_this_frame, dist_thresh=100)
        tracker_detections = unique_raw_boxes
        
        for rb in unique_raw_boxes:
            if visual_enabled:
                current_draw_boxes.append([rb[0], rb[1], rb[2], rb[3], (0, 0, 255), "Raw", 2])


        if ENABLE_TRAJECTORY_TRACKING:
            should_empty_update = (
                not got_inference_result
                and processed_for_detection
                and not sent_inference_this_frame
                and f_idx - last_tracker_update_frame >= PROCESS_EVERY_N_FRAMES
            )
            if got_inference_result or should_empty_update:
                tracker.update(tracker_detections if got_inference_result else [], frame=frame, frame_idx=f_idx)
                last_tracker_update_frame = f_idx
        if ENABLE_TRAJECTORY_TRACKING:
            confirmed_tracks = tracker.get_confirmed_tracks(f_idx)
        else:
            confirmed_tracks = []
            for idx, rb in enumerate(unique_raw_boxes):
                cx = (rb[0] + rb[2]) * 0.5
                cy = (rb[1] + rb[3]) * 0.5
                confirmed_tracks.append({
                    'id': idx + 1,
                    'box': rb[:4],
                    'history': [(cx, cy)],
                    'score': rb[4] if len(rb) > 4 else CONF_THRESH,
                    'live': True,
                    'misses': 0,
                })
        live_confirmed_tracks = [t for t in confirmed_tracks if t.get('live', True)]
        confirmed = [t['box'] for t in live_confirmed_tracks]
        for box in confirmed:
            if visual_enabled:
                current_draw_boxes.append([box[0], box[1], box[2], box[3], (0, 255, 0), "TARGET", 1])
            
        if not learning_mode:
            if confirmed:
                gimbal_data = [[b[0], b[1], b[2], b[3], estimate_rough_range_m(b, W)] for b in confirmed]
                data_sender.send_packet("data", cam_idx, gimbal_data, target="gimbal")
        if visual_enabled and len(current_draw_boxes) > MAX_DRAW_BOXES:
            current_draw_boxes = current_draw_boxes[-MAX_DRAW_BOXES:]

        if draw_this_frame:
            try:
                display_src = make_fused_display_frame(frame, latest_fusion_mask, W, H) if ENABLE_FUSION_DISPLAY else frame
                show_frame = cv2.resize(display_src, (640, 360))
                dsx, dsy = 640.0/W, 360.0/H
                for i in range(len(current_draw_boxes)-1, -1, -1):
                    x1, y1, x2, y2, color, text, life = current_draw_boxes[i]
                    cv2.rectangle(show_frame, (int(x1*dsx), int(y1*dsy)), (int(x2*dsx), int(y2*dsy)), color, 1 if text in ("ROI", "TRKROI") else 2)
                    current_draw_boxes[i][6] -= 1
                    if current_draw_boxes[i][6] <= 0: current_draw_boxes.pop(i)
                
                if send_video_this_frame: v_sender.send(BOARD_ID, cam_idx, show_frame)
                if SHOW_INDIVIDUAL_WINDOWS:
                    put_latest(display_queues[cam_idx], show_frame)
            except: pass
    cap.release()

if __name__ == '__main__':
    if SHOW_INDIVIDUAL_WINDOWS:
        for i in range(N_CAM):
            cv2.namedWindow(f"Cam {i}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"Cam {i}", 640, 360)
        latest_display_frames = [None for _ in range(N_CAM)]
    
    print("--> Detection enabled. Local mosaic display disabled.", flush=True)
    t_inf = threading.Thread(target=inference_worker); t_inf.daemon = True; t_inf.start()
    for i in range(N_CAM):
        t = threading.Thread(target=capture_job, args=(i,)); t.daemon = True; t.start()
        time.sleep(0.5)
    
    def timer_job():
        global INIT_SIGNAL_SENT, VIDEO_STREAM_ALLOWED
        while not stop_event.is_set():
            if not INIT_SIGNAL_SENT and (time.time() - SYSTEM_START_TIME > INIT_TIME):

                VIDEO_STREAM_ALLOWED = True; INIT_SIGNAL_SENT = True
            time.sleep(1)
    
    threading.Thread(target=timer_job, daemon=True).start()

    try:
        while True:
            if SHOW_INDIVIDUAL_WINDOWS:
                for i in range(N_CAM):
                    while True:
                        try:
                            latest_display_frames[i] = display_queues[i].get_nowait()
                        except queue.Empty:
                            break
                    if latest_display_frames[i] is not None:
                        cv2.imshow(f"Cam {i}", latest_display_frames[i])
                if cv2.waitKey(10) & 0xFF == ord('q'): break
            else:
                time.sleep(0.05)
    except KeyboardInterrupt: pass
    stop_event.set(); time.sleep(1.0)
    cv2.destroyAllWindows()
