"""
无人机视频分割分析脚本
处理视频前10秒，抽帧5fps，调用分割API获取mask，输出标注视频和统计JSON。

用法:
    python drone_analysis_seg.py --video ./data/test.mp4 --api_url http://your-server:port
    python drone_analysis_seg.py --video ./data/test.mp4 --api_url http://localhost:8080 --prompts "icon tower" "utility pole"
"""

import os
import cv2
import av
import json
import argparse
import base64
import time
import numpy as np
import requests
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# ─── 配置 ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR    = "./output"
INPUT_SIZE    = 416       # 保留用于兼容性
MAX_DURATION  = 10        # 最多处理前 N 秒
SAMPLE_FPS    = 5         # 抽帧率 (fps)
OUTPUT_FPS    = 5         # 输出标注视频帧率

# API 配置
API_ENDPOINT = "/BVS/GROUNDED_SEGMENT/PREDICT_BY_LABELS"
API_TIMEOUT  = 30         # 秒
API_MAX_RETRIES = 3       # 最大重试次数
API_RETRY_DELAY = 2       # 服务器错误重试延迟（秒）

# 分割参数
TEXT_PROMPTS     = ["icon tower", "utility pole", "street light"]
BOX_THRESHOLD    = 0.25
TEXT_THRESHOLD   = 0.25
NMS_THRESHOLD    = 0.8

# 切片参数（SAHI 算法）
USE_SLICING      = True      # 是否使用切片
TILE_COLS        = 4         # 横向分块数
TILE_ROWS        = 2         # 纵向分块数
TILE_OVERLAP     = 0.1       # tile 重叠比例 (10%)

# 可视化参数
MASK_ALPHA       = 0.3    # mask 填充透明度 (0.0-1.0)
BORDER_THICKNESS = 2      # polygon 边框粗细

# 颜色映射 (BGR 格式)
PROMPT_COLORS = {
    "icon tower":    (127, 255,   0),  # 亮绿色
    "utility pole":  (255, 165,   0),  # 橙色
    "street light":  (0,   80,  255),  # 红色
    # 保留其他类别作为备选
    "person":        (0,   255, 255),  # 黄色
    "car":           (255, 0,   255),  # 紫色
}


# ─── Segmentation API 客户端 ────────────────────────────────────────────────────

class SegmentationAPI:
    """分割 API HTTP 客户端"""

    def __init__(self, api_base_url, timeout=API_TIMEOUT, max_retries=API_MAX_RETRIES, retry_delay=API_RETRY_DELAY):
        """
        初始化 API 客户端

        Args:
            api_base_url: API 基础 URL (如 http://localhost:8080)
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 服务器错误重试延迟（秒）
        """
        self.base_url = api_base_url.rstrip('/')
        self.endpoint = f"{self.base_url}{API_ENDPOINT}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()

    def encode_frame(self, frame_bgr, quality=85):
        """
        将 BGR numpy 数组编码为 base64 JPEG 字符串

        Args:
            frame_bgr: OpenCV BGR 格式图像
            quality: JPEG 质量 (1-100)

        Returns:
            base64 编码的 JPEG 字符串
        """
        # 编码为 JPEG
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        # 转换为 base64
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return base64_str

    def segment_frame(self, frame_bgr, text_prompts,
                     box_thresh=BOX_THRESHOLD,
                     text_thresh=TEXT_THRESHOLD,
                     nms_thresh=NMS_THRESHOLD):
        """
        调用分割 API 处理单帧

        Args:
            frame_bgr: BGR 格式图像
            text_prompts: 文本提示列表 (如 ["person", "car"])
            box_thresh: 检测框阈值
            text_thresh: 文本阈值
            nms_thresh: NMS 阈值

        Returns:
            masks 列表，每个 mask 包含:
                - name: 类别名称
                - score: 置信度
                - area: 面积
                - mask: polygon 点数组 [[x,y], ...]
        """
        # 编码帧
        base64_image = self.encode_frame(frame_bgr)

        # 构建请求 payload
        payload = {
            "input_image": base64_image,
            "img_type": "bsae64",  # 注意：API 文档中的拼写（可能是 "base64" 的拼写错误）
            "text_prompt_list": text_prompts,
            "box_threshold": box_thresh,
            "text_threshold": text_thresh,
            "nms_threshold": nms_thresh,
            "ret_image": False,
            "save_result_image": False
        }

        # 发送请求（带重试）
        for retry in range(self.max_retries):
            try:
                response = self.session.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return self._parse_response(response.json())

                # 服务器错误（5xx）延长等待时间后重试
                elif 500 <= response.status_code < 600:
                    if retry < self.max_retries - 1:
                        # 使用更长的延迟时间（固定 retry_delay，不使用指数退避）
                        print(f"  [警告] API 服务器错误 {response.status_code}，{self.retry_delay}秒后重试 ({retry+1}/{self.max_retries})")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        print(f"  [错误] API 服务器错误 {response.status_code}，已重试 {self.max_retries} 次，跳过该帧")
                        return []

                # 客户端错误（4xx）不重试
                elif 400 <= response.status_code < 500:
                    print(f"  [警告] API 请求错误 {response.status_code}: {response.text[:100]}")
                    return []

                else:
                    print(f"  [警告] API 返回 {response.status_code}: {response.text[:200]}")
                    return []

            except requests.exceptions.Timeout:
                if retry < self.max_retries - 1:
                    wait_time = 2 ** retry  # 指数退避
                    print(f"  [警告] API 超时，{wait_time}秒后重试 ({retry+1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  [错误] API 超时，已重试 {self.max_retries} 次")

            except requests.exceptions.RequestException as e:
                if retry < self.max_retries - 1:
                    wait_time = 2 ** retry
                    print(f"  [警告] 请求失败: {e}，{wait_time}秒后重试 ({retry+1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  [错误] 请求失败: {e}")

        return []  # 所有重试失败，返回空列表

    def _parse_response(self, response_json):
        """
        解析 API 响应

        Args:
            response_json: API 返回的 JSON 数据

        Returns:
            标准化的 masks 列表
        """
        try:
            # 验证响应结构
            if "masks" not in response_json:
                print(f"  [警告] API 响应缺少 'masks' 字段: {response_json.keys()}")
                return []
            # print(f"  [调试] API 返回 :{response_json}")
            masks = response_json["masks"]
            if not isinstance(masks, list):
                print(f"  [警告] 'masks' 不是列表类型")
                return []

            # 标准化格式
            result = []
            for mask in masks:
                if not all(k in mask for k in ["name", "score", "area", "mask"]):
                    continue

                # 保持 mask 为原始列表格式，不转换为 numpy 数组
                # 因为不同 mask 的团块数量可能不同，会导致转换失败
                result.append({
                    "name": mask["name"],
                    "score": float(mask["score"]),
                    "area": int(mask["area"]),
                    "mask": mask["mask"]  # 保持原始格式
                })

            return result

        except (KeyError, ValueError, TypeError) as e:
            print(f"  [警告] 解析 API 响应失败: {e}")
            return []

    def segment_frame_with_slicing(self, frame_bgr, text_prompts,
                                   box_thresh=BOX_THRESHOLD,
                                   text_thresh=TEXT_THRESHOLD,
                                   nms_thresh=NMS_THRESHOLD,
                                   tile_cols=TILE_COLS,
                                   tile_rows=TILE_ROWS,
                                   overlap=TILE_OVERLAP):
        """
        使用切片算法（SAHI）处理帧

        Args:
            frame_bgr: BGR 格式图像
            text_prompts: 文本提示列表
            box_thresh: 检测框阈值
            text_thresh: 文本阈值
            nms_thresh: NMS 阈值
            tile_cols: 横向分块数
            tile_rows: 纵向分块数
            overlap: 重叠比例

        Returns:
            合并后的 masks 列表（全帧坐标系）
        """
        H, W = frame_bgr.shape[:2]
        base_w = W // tile_cols
        base_h = H // tile_rows
        ow = int(base_w * overlap)
        oh = int(base_h * overlap)

        all_masks = []

        # 遍历所有 tile
        for row in range(tile_rows):
            for col in range(tile_cols):
                # 计算 tile 的坐标
                x1 = max(0, col * base_w - ow)
                y1 = max(0, row * base_h - oh)
                x2 = min(W, (col + 1) * base_w + ow)
                y2 = min(H, (row + 1) * base_h + oh)

                # 提取 tile
                tile = frame_bgr[y1:y2, x1:x2]

                # 调用 API 处理该 tile
                tile_masks = self.segment_frame(
                    tile, text_prompts,
                    box_thresh, text_thresh, nms_thresh
                )

                # 将 tile 坐标系转换回全帧坐标系
                for mask in tile_masks:
                    adjusted_mask = self._adjust_mask_coordinates(
                        mask, x1, y1
                    )
                    all_masks.append(adjusted_mask)

        # 合并重复的 masks（基于 IoU）
        merged_masks = self._merge_overlapping_masks(all_masks, nms_thresh)

        return merged_masks

    def _adjust_mask_coordinates(self, mask, offset_x, offset_y):
        """
        将 tile 坐标系下的 mask 转换到全帧坐标系

        Args:
            mask: mask 数据
            offset_x: x 方向偏移
            offset_y: y 方向偏移

        Returns:
            调整后的 mask
        """
        adjusted = {
            "name": mask["name"],
            "score": mask["score"],
            "area": mask["area"],
            "mask": []
        }

        # 遍历所有团块（contours）
        for contour in mask["mask"]:
            # contour 可能是 [[x,y], ...] 或嵌套结构
            try:
                contour_array = np.array(contour, dtype=np.int32)

                # 对每个点添加偏移
                if len(contour_array.shape) == 2 and contour_array.shape[1] == 2:
                    # [[x,y], ...] 格式
                    contour_array[:, 0] += offset_x
                    contour_array[:, 1] += offset_y
                    adjusted["mask"].append(contour_array.tolist())
                else:
                    # 其他格式，保持原样
                    adjusted["mask"].append(contour)
            except (ValueError, IndexError):
                # 转换失败，保持原样
                adjusted["mask"].append(contour)

        return adjusted

    def _merge_overlapping_masks(self, masks, iou_threshold=0.5):
        """
        合并重叠的 masks（基于简化的 IoU 计算）

        Args:
            masks: mask 列表
            iou_threshold: IoU 阈值

        Returns:
            合并后的 mask 列表
        """
        if len(masks) == 0:
            return []

        # 计算每个 mask 的边界框
        mask_boxes = []
        for mask in masks:
            boxes = []
            for contour in mask["mask"]:
                try:
                    contour_array = np.array(contour, dtype=np.int32)
                    if len(contour_array.shape) == 2 and contour_array.shape[1] == 2:
                        x_min = int(np.min(contour_array[:, 0]))
                        y_min = int(np.min(contour_array[:, 1]))
                        x_max = int(np.max(contour_array[:, 0]))
                        y_max = int(np.max(contour_array[:, 1]))
                        boxes.append([x_min, y_min, x_max, y_max])
                except (ValueError, IndexError):
                    continue

            # 如果有多个团块，计算总的边界框
            if boxes:
                all_boxes = np.array(boxes)
                x_min = int(np.min(all_boxes[:, 0]))
                y_min = int(np.min(all_boxes[:, 1]))
                x_max = int(np.max(all_boxes[:, 2]))
                y_max = int(np.max(all_boxes[:, 3]))
                mask_boxes.append([x_min, y_min, x_max, y_max])
            else:
                # 没有有效的团块，使用默认值
                mask_boxes.append([0, 0, 0, 0])

        mask_boxes = np.array(mask_boxes)

        # 使用 NMS 合并重叠的 masks
        keep_indices = self._nms_masks(mask_boxes, [m["score"] for m in masks], iou_threshold)

        # 确保 keep_indices 是列表类型
        return [masks[int(i)] for i in keep_indices]

    def _nms_masks(self, boxes, scores, iou_threshold):
        """
        对 masks 进行 NMS

        Args:
            boxes: 边界框数组 [[x1,y1,x2,y2], ...]
            scores: 置信度列表
            iou_threshold: IoU 阈值

        Returns:
            保留的索引列表
        """
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)

        # 按置信度降序排序
        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            # 保留当前最高分的框
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # 计算当前框与其他框的 IoU
            ious = self._compute_iou(boxes[current], boxes[indices[1:]])

            # 保留 IoU 小于阈值的框
            indices = indices[1:][ious < iou_threshold]

        return keep

    def _compute_iou(self, box1, box2):
        """
        计算两个边界框的 IoU

        Args:
            box1: [x1, y1, x2, y2]
            box2: [[x1, y1, x2, y2], ...] 或 [x1, y1, x2, y2]

        Returns:
            IoU 数组或单个 IoU 值
        """
        box1 = np.array(box1)
        box2 = np.array(box2)

        if len(box2.shape) == 1:
            box2 = box2.reshape(1, -1)

        # 计算交集
        x1 = np.maximum(box1[0], box2[:, 0])
        y1 = np.maximum(box1[1], box2[:, 1])
        x2 = np.minimum(box1[2], box2[:, 2])
        y2 = np.minimum(box1[3], box2[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # 计算并集
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = area1 + area2 - intersection

        # 计算 IoU
        iou = intersection / (union + 1e-6)

        if len(iou) == 1:
            return iou[0]
        return iou


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


# ─── 绘制 Mask ─────────────────────────────────────────────────────────────────

def _compute_centroid(polygon):
    """
    计算 polygon 的质心

    Args:
        polygon: Nx2 numpy array, [[x1,y1], [x2,y2], ...]

    Returns:
        (cx, cy) 质心坐标
    """
    if len(polygon) == 0:
        return (0, 0)
    # 简单平均所有点
    cx = int(np.mean(polygon[:, 0]))
    cy = int(np.mean(polygon[:, 1]))
    return (cx, cy)


def _get_mask_color(name):
    """获取类别对应的颜色"""
    return PROMPT_COLORS.get(name, (0, 255, 255))  # 默认黄色


def draw_masks(frame_bgr, masks, timestamp):
    """
    在帧上绘制分割 masks

    Args:
        frame_bgr: BGR 格式图像
        masks: mask 列表（来自 API）
        timestamp: 时间戳（秒）

    Returns:
        绘制后的 BGR 图像
    """
    img   = frame_bgr.copy()
    H, W  = img.shape[:2]
    thick = max(1, int(0.6 * (H + W) / 1000))

    # 创建半透明 overlay
    overlay = img.copy()

    for mask in masks:
        name  = mask["name"]
        score = mask["score"]
        mask_data = mask["mask"]

        # mask_data 是列表格式，可能是：
        # 1. 多个团块：[[[x,y], ...], [[x,y], ...], ...] 或 [[[x,y], ...]], [[[x,y], ...]]
        # 2. 单个团块：[[x,y], ...]

        # 检查是否为空
        if not mask_data or len(mask_data) == 0:
            continue

        color = _get_mask_color(name)
        all_contours = []

        # 尝试识别数据结构
        try:
            # 检查第一个元素的结构
            first_element = mask_data[0]

            # 如果第一个元素是列表且包含数字对 -> 单个团块 [[x,y], ...]
            if isinstance(first_element, list) and len(first_element) > 0:
                if isinstance(first_element[0], list):
                    # [[[x,y], ...], ...] 或 [[[[x,y], ...]], [[[x,y], ...]]]
                    # 多个团块的情况
                    for blob in mask_data:
                        if isinstance(blob, list):
                            # 可能是 [[[x,y], ...]] 这种嵌套，需要展平
                            if len(blob) > 0 and isinstance(blob[0], list):
                                # 检查是否是 [[[x,y], ...]] 这种结构
                                if len(blob[0]) > 0 and isinstance(blob[0][0], list):
                                    # 需要再展开一层
                                    for nested in blob:
                                        if len(nested) >= 3:
                                            contour = np.array(nested, dtype=np.int32)
                                            all_contours.append(contour)
                                elif len(blob) >= 3:
                                    # 正常的 [[x,y], ...] 结构
                                    contour = np.array(blob, dtype=np.int32)
                                    all_contours.append(contour)
                        elif len(blob) >= 3:
                            # blob 本身就是 [[x,y], ...] 结构
                            contour = np.array(blob, dtype=np.int32)
                            all_contours.append(contour)
                else:
                    # 单个团块 [[x,y], ...]
                    if len(mask_data) >= 3:
                        contour = np.array(mask_data, dtype=np.int32)
                        all_contours.append(contour)

            if not all_contours:
                continue

            # 绘制所有团块
            for contour in all_contours:
                cv2.fillPoly(overlay, [contour], color)
                cv2.polylines(img, [contour], True, color, thick)

            # 计算质心（使用所有团块的点）
            all_contour_points = np.vstack(all_contours)
            cx, cy = _compute_centroid(all_contour_points)

        except (IndexError, TypeError, ValueError) as e:
            print(f"  [警告] 跳过无效的 mask 数据: {e}")
            continue

        # 标签背景 + 中文文字
        label = f"{name} {score:.2f}"
        bbox  = _FONT_LABEL.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 确保标签不超出图像边界
        label_x1 = max(0, cx - tw // 2)
        label_y1 = max(0, cy - th - 8)
        label_x2 = min(W, label_x1 + tw + 4)
        label_y2 = max(0, cy)

        cv2.rectangle(img, (label_x1, label_y1), (label_x2, label_y2), color, -1)
        img = _put_cn_text(img, label, (label_x1 + 2, label_y1 + 2), (0, 0, 0), _FONT_LABEL)

    # 混合 overlay 实现半透明效果
    img = cv2.addWeighted(overlay, MASK_ALPHA, img, 1 - MASK_ALPHA, 0)

    # 右上角水印
    cv2.putText(img, f"t={timestamp:.1f}s", (W - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(img, "SEGMENT 5fps/10s", (W - 230, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # cv2.namedWindow(f"Segmentation -", cv2.WINDOW_NORMAL)  
    # cv2.imshow(f"Segmentation -", img)
    # cv2.waitKey(0)

    return img


# ─── 主处理函数 ────────────────────────────────────────────────────────────────

def process_video(video_path, video_id=None, output_dir=OUTPUT_DIR,
                  api_base_url=None, prompts=None, max_duration=MAX_DURATION,
                  sample_fps=SAMPLE_FPS, output_fps=OUTPUT_FPS,
                  use_slicing=False, tile_cols=TILE_COLS, tile_rows=TILE_ROWS, tile_overlap=TILE_OVERLAP):
    """
    处理视频：调用分割 API，绘制 masks，输出标注视频

    Args:
        video_path: 输入视频路径
        video_id: 视频 ID（默认取文件名）
        output_dir: 输出目录
        api_base_url: API 基础 URL
        prompts: 文本提示列表
        max_duration: 处理前 N 秒
        sample_fps: 采样帧率
        output_fps: 输出视频帧率
        use_slicing: 是否使用切片模式
        tile_cols: 横向分块数
        tile_rows: 纵向分块数
        tile_overlap: tile 重叠比例

    Returns:
        统计信息字典
    """
    if api_base_url is None:
        raise ValueError("必须提供 --api_url 参数")

    if prompts is None:
        prompts = TEXT_PROMPTS

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
    max_frame       = min(int(src_fps * max_duration), total_frames)
    sample_interval = max(1, round(src_fps / sample_fps))

    print(f"[INFO] 视频: {video_path}  分辨率: {W}×{H}  源FPS: {src_fps:.1f}")
    print(f"[INFO] 处理前 {max_duration}s ({max_frame} 帧)，每 {sample_interval} 帧取 1 帧")
    print(f"[INFO] API: {api_base_url}")
    print(f"[INFO] 分割类别: {prompts}")
    if use_slicing:
        print(f"[INFO] 切片模式: {tile_cols}×{tile_rows} 重叠={int(tile_overlap*100)}%")

    out_video_path = os.path.join(output_dir, f"{video_id}_seg_annotated.mp4")

    # 用 PyAV 写 H.264
    av_output = av.open(out_video_path, 'w')
    av_stream = av_output.add_stream('h264', rate=output_fps)
    av_stream.width  = W
    av_stream.height = H
    av_stream.pix_fmt = 'yuv420p'
    av_stream.options = {'crf': '23', 'preset': 'fast'}

    # 初始化 API 客户端
    api = SegmentationAPI(api_base_url)

    total_counts = {}
    per_frame    = []
    frame_idx    = 0
    sampled      = 0

    print(f"\n[开始处理]")
    first_frame = True

    while frame_idx < max_frame:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            timestamp = frame_idx / src_fps

            # 调用分割 API（根据配置选择切片或全帧）
            if use_slicing:
                masks = api.segment_frame_with_slicing(
                    frame_bgr, prompts,
                    BOX_THRESHOLD, TEXT_THRESHOLD, NMS_THRESHOLD,
                    tile_cols, tile_rows, tile_overlap
                )
            else:
                masks = api.segment_frame(
                    frame_bgr, prompts,
                    BOX_THRESHOLD, TEXT_THRESHOLD, NMS_THRESHOLD
                )

            # 逐帧统计
            frame_counts = {}
            for mask in masks:
                name = mask["name"]
                frame_counts[name] = frame_counts.get(name, 0) + 1
                total_counts[name] = total_counts.get(name, 0) + 1

            per_frame.append({
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 2),
                "counts": frame_counts,
                "num_masks": len(masks),
            })

            # 绘制 masks
            annotated = draw_masks(frame_bgr, masks, timestamp)
            # cv2.imshow(f"Segmentation - {video_id}", annotated)
            # cv2.waitKey(1)
            cv2.imwrite(f"{video_id}_seg_frame_{frame_idx:05d}.jpg", annotated)

            # 写入视频
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(annotated_rgb, format='rgb24')
            av_frame = av_frame.reformat(format='yuv420p')
            for pkt in av_stream.encode(av_frame):
                av_output.mux(pkt)

            # 第一帧显示 API 信息
            if first_frame:
                print(f"  [{sampled:3d}] t={timestamp:5.1f}s  masks={len(masks):2d}  {frame_counts}  [API 连接成功]")
                first_frame = False
            else:
                print(f"  [{sampled:3d}] t={timestamp:5.1f}s  masks={len(masks):2d}  {frame_counts}")

            sampled += 1
            time.sleep(0.5)

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
        "duration_processed_s": max_duration,
        "sample_fps":           sample_fps,
        "frames_processed":     sampled,
        "annotated_video":      f"{video_id}_seg_annotated.mp4",
        "api_endpoint":         f"{api_base_url}{API_ENDPOINT}",
        "text_prompts":         prompts,
        "thresholds": {
            "box": BOX_THRESHOLD,
            "text": TEXT_THRESHOLD,
            "nms": NMS_THRESHOLD
        },
        "totals":               total_counts,
        "per_frame":            per_frame,
    }

    stats_path = os.path.join(output_dir, f"{video_id}_seg_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 标注视频 → {out_video_path}")
    print(f"[完成] 统计JSON → {stats_path}")
    print(f"[汇总] {total_counts}")
    return stats


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="无人机视频分割分析 - 调用分割API获取mask并绘制到视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python drone_analysis_seg.py --video ./data/test.mp4 --api_url http://localhost:8000
  python drone_analysis_seg.py --video ./data/test.mp4 --api_url http://192.168.1.154:8000 --prompts "icon tower" "utility pole"
  python drone_analysis_seg.py --video ./data/test.mp4 --api_url http://localhost:8000 --duration 5
        """
    )
    parser.add_argument("--video",      required=True,       help="输入视频路径")
    parser.add_argument("--api_url",    required=True,       help="分割API基础URL (如: http://localhost:8000)")
    # parser.add_argument("--video",      default=r"V:\datasets\2026\4K_video\DJI_20260125104749_0001_V.MP4",       help="输入视频路径")
    # parser.add_argument("--api_url",    default="http://localhost:8000",       help="分割API基础URL (如: http://localhost:8000)")
    parser.add_argument("--video_id",   default=None,        help="视频ID（默认取文件名）")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,  help="输出目录")
    parser.add_argument("--prompts",    nargs="+",           help="分割类别列表 (如: 'icon tower' 'utility pole' 'street light')", default=None)
    parser.add_argument("--duration",   type=int,            help="处理前N秒 (默认: 10)", default=MAX_DURATION)
    parser.add_argument("--sample_fps", type=int,            help="采样帧率 (默认: 5)", default=SAMPLE_FPS)
    # 切片参数
    parser.add_argument("--use_slicing", action="store_true", help="启用切片模式（SAHI算法）")
    parser.add_argument("--tile_cols",  type=int,            help="横向分块数 (默认: 4)", default=TILE_COLS)
    parser.add_argument("--tile_rows",  type=int,            help="纵向分块数 (默认: 2)", default=TILE_ROWS)
    parser.add_argument("--tile_overlap", type=float,        help="tile重叠比例 (默认: 0.1)", default=TILE_OVERLAP)
    args = parser.parse_args()

    # 命令行参数覆盖配置
    use_slicing = args.use_slicing
    tile_cols = args.tile_cols
    tile_rows = args.tile_rows
    tile_overlap = args.tile_overlap

    process_video(
        video_path=args.video,
        video_id=args.video_id,
        output_dir=args.output_dir,
        api_base_url=args.api_url,
        prompts=args.prompts,
        max_duration=args.duration,
        sample_fps=args.sample_fps,
        use_slicing=use_slicing,
        tile_cols=tile_cols,
        tile_rows=tile_rows,
        tile_overlap=tile_overlap,
    )
