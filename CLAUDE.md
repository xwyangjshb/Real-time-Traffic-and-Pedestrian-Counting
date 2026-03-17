# Real-time Traffic and Pedestrian Counting

YOLOv3-based object detection with SORT multi-object tracking and line-crossing counting.

## Stack

- Python 3.6, TensorFlow 2.0 (GPU), OpenCV
- CUDA 10.0, cuDNN compatible
- SORT tracker (Kalman filter + Hungarian algorithm)

## Setup

```bash
conda env create -f environment.yml
conda activate <env-name>

# Download YOLOv3 weights (darknet format) and place at path set in core/config.py (cfg.YOLO.WEIGHTS)
```

## Run Commands

```bash
# Webcam/video inference with counting
python video_demo.py

# Single image inference
python image_demo.py

# Training
python train.py

# Evaluation (mAP)
python test.py
```

## Configuration

Edit these variables directly in `video_demo.py` before running:

| Variable | Description | Example |
|---|---|---|
| `video_path` | Video file path or `0` for webcam | `"./data/video.mp4"` or `0` |
| `specified_class_id_filter` | COCO class IDs to detect/count | `[2]` = car, `[0]` = person, `[0,2]` = both |
| `line` | Counting line as two (x,y) points | `[(0,400),(1280,400)]` |

Global config (thresholds, anchors, training params) lives in `core/config.py` via EasyDict.
Key fields: `cfg.YOLO.WEIGHTS`, `cfg.YOLO.IOU_LOSS_THRESH`, `cfg.TRAIN.*`.

## Architecture

```
video_demo.py
  └─ loads YOLOv3 (core/yolov3.py → core/backbone.py → core/common.py)
  └─ per-frame: detect → decode → NMS  (core/utils.py)
  └─ SORT tracker updates              (core/sort.py)
  └─ line-crossing count               (core/utils.py: video_draw_bbox)
```

### Pipeline detail

1. **Detection** — Darknet-53 backbone (`core/backbone.py`) → YOLOv3 head (`core/yolov3.py`) outputs boxes at 3 scales
2. **Decoding/NMS** — `core/utils.py` decodes raw predictions, applies NMS
3. **Tracking** — `core/sort.py`: `KalmanBoxTracker` per object, `Sort.update()` uses Hungarian matching each frame; each track gets a persistent ID
4. **Counting** — `video_draw_bbox` in `core/utils.py` checks whether a track's centroid crosses the configured `line` between frames; increments `count_up` / `count_down`

## Key Files

| File | Role |
|---|---|
| `video_demo.py` | Main entry point — edit runtime config here |
| `core/utils.py` | Weight loading, bbox drawing, SORT glue, **counting logic** (`video_draw_bbox`) |
| `core/sort.py` | SORT tracker (`KalmanBoxTracker`, `Sort`) |
| `core/yolov3.py` | YOLOv3 model, decode, loss |
| `core/backbone.py` | Darknet-53 feature extractor |
| `core/common.py` | Conv/BN/residual building blocks |
| `core/config.py` | EasyDict config — thresholds, anchors, paths |
| `core/dataset.py` | Data loading + augmentation for training |

## Notes

- COCO class IDs: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
- Counting direction: objects crossing the line top-to-bottom vs bottom-to-top are counted separately (`count_up` / `count_down`)
- TF 2.0 + CUDA 10.0 is a strict requirement; newer CUDA versions require dependency changes
