"""
无人机视频离线分析脚本
处理视频前30秒，抽帧1fps，4K分块检测，输出标注视频和统计JSON。

用法:
    python drone_analysis.py --video ./data/test.mp4
    python drone_analysis.py --video ./data/test.mp4 --video_id demo_001
"""

import os
import cv2
import av
import json
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

import core.utils as utils
from core.yolov3 import YOLOv3, decode


# ─── 配置 ──────────────────────────────────────────────────────────────────────

WEIGHTS_PATH  = "./yolov3.weights"
OUTPUT_DIR    = "./output"
INPUT_SIZE    = 416       # YOLOv3 模型输入尺寸
MAX_DURATION  = 10        # 最多处理前 N 秒
SAMPLE_FPS    = 5         # 抽帧率 (fps)
OUTPUT_FPS    = 5         # 输出标注视频帧率
TILE_COLS     = 4         # 横向分块数
TILE_ROWS     = 2         # 纵向分块数
TILE_OVERLAP  = 0.1       # tile 重叠比例 (10%)
SCORE_THRESH  = 0.25      # 检测置信度阈值
NMS_THRESH    = 0.45      # NMS IOU 阈值

# 目标类别 (当前使用 COCO 类别作为占位符，实际部署时替换为自定义模型的类别 ID)
TARGET_CLASSES = {
    0: "电线杆",   # COCO: person  (占位符)
    2: "铁塔",     # COCO: car     (占位符)
    7: "光纤标识", # COCO: truck   (占位符)
}

# BGR 颜色
CLASS_COLORS = {
    "电线杆":   (127, 255, 0),
    "铁塔":     (255, 165, 0),
    "光纤标识": (0,   80,  255),
}


# ─── 模型 ──────────────────────────────────────────────────────────────────────

def build_model(weights_path, input_size):
    input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv3(input_layer)
    bbox_tensors = [
        tf.keras.layers.Lambda(lambda x, j=i: decode(x, j), name=f'decode_{i}')(fm)
        for i, fm in enumerate(feature_maps)
    ]
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, weights_path)
    print(f"[INFO] 模型加载完成: {weights_path}")
    return model


# ─── 分块检测 ──────────────────────────────────────────────────────────────────

def _detect_tile(model, tile_rgb, input_size, score_thresh, nms_thresh):
    """对单个 tile (RGB) 做推理，返回 tile 坐标系下的 bbox 数组。"""
    th, tw = tile_rgb.shape[:2]
    image_data = utils.image_preporcess(np.copy(tile_rgb), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred = model.predict_on_batch(image_data)
    pred = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred]
    pred = tf.concat(pred, axis=0)

    bboxes = utils.postprocess_boxes(pred, (th, tw), input_size, score_thresh)
    if len(bboxes) > 0:
        bboxes = utils.nms(bboxes, nms_thresh, method='nms')
    return bboxes  # list or ndarray


def tile_detect(model, frame_rgb, input_size,
                tile_cols, tile_rows, overlap,
                score_thresh, nms_thresh):
    """将帧分块检测后合并全帧 NMS，返回全帧坐标系下的 bbox 列表。"""
    H, W = frame_rgb.shape[:2]
    base_w = W // tile_cols
    base_h = H // tile_rows
    ow = int(base_w * overlap)
    oh = int(base_h * overlap)

    all_boxes = []
    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = max(0, col * base_w - ow)
            y1 = max(0, row * base_h - oh)
            x2 = min(W, (col + 1) * base_w + ow)
            y2 = min(H, (row + 1) * base_h + oh)

            tile = frame_rgb[y1:y2, x1:x2]
            bboxes = _detect_tile(model, tile, input_size, score_thresh, nms_thresh)

            if len(bboxes) > 0:
                bboxes = np.array(bboxes)
                bboxes[:, 0] += x1
                bboxes[:, 2] += x1
                bboxes[:, 1] += y1
                bboxes[:, 3] += y1
                all_boxes.append(bboxes)

    if not all_boxes:
        return []

    all_boxes = np.vstack(all_boxes)
    merged = utils.nms(all_boxes, nms_thresh, method='nms')
    return merged


# ─── 中文字体 ──────────────────────────────────────────────────────────────────

def _load_font(size):
    """按优先级尝试加载中文字体，全部失败则用 PIL 默认字体。"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

_FONT_LABEL  = _load_font(28)
_FONT_WATER  = _load_font(32)


def _put_cn_text(img_bgr, text, pos, color_bgr, font):
    """在 BGR OpenCV 图像上用 PIL 渲染中文，返回 BGR 图像。"""
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(img_rgb)
    draw     = ImageDraw.Draw(pil_img)
    # PIL 用 RGB
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ─── 绘制标注 ──────────────────────────────────────────────────────────────────

def draw_detections(frame_bgr, bboxes, timestamp):
    img   = frame_bgr.copy()
    H, W  = img.shape[:2]
    thick = max(1, int(0.6 * (H + W) / 1000))

    for b in bboxes:
        cls_id = int(b[5])
        if cls_id not in TARGET_CLASSES:
            continue
        name  = TARGET_CLASSES[cls_id]
        color = CLASS_COLORS.get(name, (0, 255, 255))
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        score = b[4]

        # 检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

        # 标签背景 + 中文文字
        label = f"{name} {score:.2f}"
        bbox  = _FONT_LABEL.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        img = _put_cn_text(img, label, (x1 + 2, y1 - th - 6), (0, 0, 0), _FONT_LABEL)

    # 右上角水印（纯 ASCII，OpenCV 即可）
    cv2.putText(img, f"t={timestamp:.1f}s", (W - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(img, "DEMO  5fps/10s", (W - 230, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    return img


# ─── 主处理函数 ────────────────────────────────────────────────────────────────

def process_video(video_path, video_id=None, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    if video_id is None:
        video_id = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frame       = min(int(src_fps * MAX_DURATION), total_frames)
    sample_interval = max(1, round(src_fps / SAMPLE_FPS))

    print(f"[INFO] 视频: {video_path}  分辨率: {W}×{H}  源FPS: {src_fps:.1f}")
    print(f"[INFO] 处理前 {MAX_DURATION}s ({max_frame} 帧)，每 {sample_interval} 帧取 1 帧")

    out_video_path = os.path.join(output_dir, f"{video_id}_annotated.mp4")

    # 用 PyAV 写 H.264，保证浏览器可直接播放
    av_output = av.open(out_video_path, 'w')
    av_stream = av_output.add_stream('h264', rate=OUTPUT_FPS)
    av_stream.width  = W
    av_stream.height = H
    av_stream.pix_fmt = 'yuv420p'
    av_stream.options = {'crf': '23', 'preset': 'fast'}

    model = build_model(WEIGHTS_PATH, INPUT_SIZE)

    total_counts = {name: 0 for name in TARGET_CLASSES.values()}
    per_frame    = []
    frame_idx    = 0
    sampled      = 0

    while frame_idx < max_frame:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            timestamp = frame_idx / src_fps
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            bboxes = tile_detect(
                model, frame_rgb, INPUT_SIZE,
                TILE_COLS, TILE_ROWS, TILE_OVERLAP,
                SCORE_THRESH, NMS_THRESH
            )

            # 只保留目标类别
            if len(bboxes) > 0:
                bboxes = [b for b in bboxes if int(b[5]) in TARGET_CLASSES]

            # 逐帧统计
            frame_counts = {name: 0 for name in TARGET_CLASSES.values()}
            for b in bboxes:
                name = TARGET_CLASSES[int(b[5])]
                frame_counts[name] += 1
                total_counts[name] += 1

            per_frame.append({
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 2),
                "counts": frame_counts,
            })

            annotated = draw_detections(frame_bgr, bboxes, timestamp)
            # BGR → RGB → PyAV VideoFrame → H.264
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(annotated_rgb, format='rgb24')
            av_frame = av_frame.reformat(format='yuv420p')
            for pkt in av_stream.encode(av_frame):
                av_output.mux(pkt)

            sampled += 1
            print(f"  [{sampled:3d}] t={timestamp:5.1f}s  {frame_counts}")

        frame_idx += 1

    # 刷新剩余帧
    for pkt in av_stream.encode():
        av_output.mux(pkt)
    av_output.close()
    cap.release()

    stats = {
        "video_id":             video_id,
        "source_path":          video_path,
        "processed_at":         datetime.now().isoformat(timespec='seconds'),
        "resolution":           f"{W}x{H}",
        "source_fps":           round(src_fps, 2),
        "duration_processed_s": MAX_DURATION,
        "sample_fps":           SAMPLE_FPS,
        "frames_processed":     sampled,
        "annotated_video":      f"{video_id}_annotated.mp4",
        "totals":               total_counts,
        "per_frame":            per_frame,
    }

    stats_path = os.path.join(output_dir, f"{video_id}_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 标注视频 → {out_video_path}")
    print(f"[完成] 统计JSON → {stats_path}")
    print(f"[汇总] {total_counts}")
    return stats


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="无人机视频离线分析")
    parser.add_argument("--video",      required=True,       help="输入视频路径")
    parser.add_argument("--video_id",   default=None,        help="视频ID（默认取文件名）")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,  help="输出目录")
    args = parser.parse_args()
    process_video(args.video, args.video_id, args.output_dir)
