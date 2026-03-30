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
from translate import Translator
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
TEXT_PROMPTS     = ["icon tower", "utility pole", "fiber"]
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

# 英中类别名称映射
CLASS_NAME_ZH = {
    # "icon tower":    "铁塔",
    # "utility pole":  "电线杆",
    "street light":  "路灯",
    # "fiber":         "光纤",
    "person":        "行人",
    "car":           "汽车",
    "bicycle":       "自行车",
    "motorcycle":    "摩托车",
    "bus":           "公交车",
    "truck":         "卡车",
}

# 反向映射：中文名称 → 英文名称
CLASS_NAME_EN = {v: k for k, v in CLASS_NAME_ZH.items()}

# 颜色映射 (BGR 格式)
PROMPT_COLORS = {
    # 中文类别（主要使用）
    "铁塔":         (0,   255,   0),  # 亮绿色 - 铁塔
    "电线杆":       (255, 165,   0),  # 橙色 - 电线杆
    "路灯":         (0,   80,  255),  # 红色 - 路灯
    "光纤":         (255, 0,   0),    # 蓝色 - 光纤
    "房屋":         (255, 0,   255),  # 紫色 - 房屋
    "烟囱":         (0,   255, 255),  # 黄色 - 烟囱
    "行人":         (203, 192, 255),  # 粉色 - 行人
    "汽车":         (128, 0,   128),  # 深紫色 - 汽车
    "自行车":       (255, 105, 180),  # 粉红色 - 自行车
    "摩托车":       (139, 0,   139),  # 深洋红 - 摩托车
    "公交车":       (255, 191,   0),  # 深金色 - 公交车
    "卡车":         (0,   128, 128),  # 青色 - 卡车
    # 英文类别（保留兼容）
    "icon tower":   (0,   255,   0),  # 亮绿色
    "utility pole": (255, 165,   0),  # 橙色
    "street light": (0,   80,  255),  # 红色
    "fiber":        (255, 0,   0),    # 蓝色
    "person":       (203, 192, 255),  # 粉色
    "car":          (128, 0,   128),  # 深紫色
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
            # print(f"  [调试] API 返回 :{response_json}")
            # 验证响应结构
            if "masks" not in response_json:
                print(f"  [警告] API 响应缺少 'masks' 字段: {response_json.keys()}")
                return []
            
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
    """
    获取类别对应的颜色（支持动态生成）

    优先使用预定义颜色，如果未定义则基于类别名称哈希生成独特颜色

    Args:
        name: 类别名称

    Returns:
        tuple: BGR 格式颜色 (b, g, r)
    """
    # 1. 优先使用预定义颜色
    if name in PROMPT_COLORS:
        return PROMPT_COLORS[name]

    # 2. 基于类别名称生成独特的颜色
    # 使用哈希确保相同的类别名称总是得到相同的颜色
    import hashlib
    hash_obj = hashlib.md5(name.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()

    # 从哈希值生成RGB，确保颜色鲜艳且可见
    # 使用 HSV 转 RGB 的思路生成鲜艳的颜色
    r = int(hash_hex[0:2], 16) % 200 + 55  # 55-255
    g = int(hash_hex[2:4], 16) % 200 + 55  # 55-255
    b = int(hash_hex[4:6], 16) % 200 + 55  # 55-255

    # 返回 BGR 格式
    return (b, g, r)


def _get_chinese_name(name_en):
    """
    将英文类别名转换为中文

    Args:
        name_en: 英文类别名

    Returns:
        中文名称，如果未找到则返回原英文名
    """
    return CLASS_NAME_ZH.get(name_en, name_en)


def _get_chinese_name_with_map(name_en, en_to_zh_map=None):
    """
    将英文类别名转换为中文（优先使用映射表）

    Args:
        name_en: 英文类别名
        en_to_zh_map: 英文→中文 映射表（优先级高于字典）

    Returns:
        中文名称，如果未找到则返回原英文名
    """
    # 1. 优先使用用户提供的映射表（来自用户输入）
    if en_to_zh_map and name_en in en_to_zh_map:
        return en_to_zh_map[name_en]

    # 2. 使用本地字典
    return CLASS_NAME_ZH.get(name_en, name_en)


def _contains_chinese(text):
    """检测字符串是否包含中文字符"""
    if not text or not isinstance(text, str):
        return False
    return any('\u4e00' <= char <= '\u9fff' for char in text)


def _translate_to_english(text_zh):
    """
    将中文翻译为英文（优先使用本地字典）

    Args:
        text_zh: 中文文本

    Returns:
        str: 英文翻译，失败则返回原输入
    """
    # 1. 优先使用本地字典
    from_local = CLASS_NAME_EN.get(text_zh)
    if from_local:
        return from_local

    # 2. 使用 translate 库
    try:
        translator = Translator(from_lang="zh", to_lang="en")
        translation = translator.translate(text_zh)
        if translation and translation != text_zh:
            return translation
        else:
            print(f"  [警告] 翻译失败，使用原输入")
            return text_zh
    except Exception as e:
        print(f"  [错误] 翻译异常: {e}，使用原输入")
        return text_zh


def _translate_to_chinese(text_en):
    """
    将英文翻译为中文（优先使用本地字典）

    Args:
        text_en: 英文文本

    Returns:
        str: 中文翻译，失败则返回原输入
    """
    # 1. 优先使用本地字典
    from_local = CLASS_NAME_ZH.get(text_en)
    if from_local:
        return from_local

    # 2. 使用 translate 库
    try:
        translator = Translator(from_lang="en", to_lang="zh")
        translation = translator.translate(text_en)
        if translation and translation != text_en:
            return translation
        else:
            print(f"  [警告] 翻译失败，使用原输入")
            return text_en
    except Exception as e:
        print(f"  [错误] 翻译异常: {e}，使用原输入")
        return text_en


def _normalize_prompts(prompts):
    """
    标准化提示词列表：统一翻译为（英文用于API，中文用于显示）

    Args:
        prompts: 提示词列表，可能包含中文或英文

    Returns:
        tuple: (英文提示词列表, 中文提示词列表, 英文→中文映射表)
        - 英文列表: 用于发送给 API
        - 中文列表: 用于显示和 JSON 保存
        - 映射表: API返回的英文 → 用户期望的中文

    Example:
        输入: ["铁塔", "utility pole", "建筑物"]
        输出: (["icon tower", "utility pole", "building"],
               ["铁塔", "电线杆", "建筑物"],
               {"icon tower": "铁塔", "utility pole": "电线杆", "building": "建筑物"})
    """
    if prompts is None:
        return None, None, {}

    english_prompts = []
    chinese_prompts = []
    en_to_zh_map = {}  # 英文 → 中文 映射表

    print(f"\n[翻译处理] 标准化提示词...")

    for prompt in prompts:
        if _contains_chinese(prompt):
            # 输入是中文：翻译为英文给 API，保留中文用于显示
            en = _translate_to_english(prompt)
            english_prompts.append(en)
            chinese_prompts.append(prompt)  # 保留原始中文
            en_to_zh_map[en] = prompt  # 记录映射关系
            if en != prompt:
                print(f"  [翻译] '{prompt}' → '{en}' (API调用)")
        else:
            # 输入是英文：直接给 API，翻译为中文用于显示
            zh = _translate_to_chinese(prompt)
            english_prompts.append(prompt)  # 保留原始英文
            chinese_prompts.append(zh)
            en_to_zh_map[prompt] = zh  # 记录映射关系
            if zh != prompt:
                print(f"  [翻译] '{prompt}' → '{zh}' (显示)")
            else:
                print(f"  [保留] '{prompt}' (未找到中文映射)")

    return english_prompts, chinese_prompts, en_to_zh_map


def draw_masks_with_map(frame_bgr, masks, timestamp, en_to_zh_map=None):
    """
    在帧上绘制分割 masks（支持自定义英文→中文映射）

    Args:
        frame_bgr: BGR 格式图像
        masks: mask 列表（来自 API）
        timestamp: 时间戳（秒）
        en_to_zh_map: 英文→中文 映射表（可选）

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

        # 标签背景 + 中文文字（使用映射表）
        name_zh = _get_chinese_name_with_map(name, en_to_zh_map)
        label = f"{name_zh} {score:.2f}"
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
        name_zh = _get_chinese_name(name)
        label = f"{name_zh} {score:.2f}"
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
                  use_slicing=False, tile_cols=TILE_COLS, tile_rows=TILE_ROWS, tile_overlap=TILE_OVERLAP,
                  split_segments=False):
    """
    处理视频：调用分割 API，绘制 masks，输出标注视频

    Args:
        video_path: 输入视频路径
        video_id: 视频 ID（默认取文件名）
        output_dir: 输出目录
        api_base_url: API 基础 URL
        prompts: 文本提示列表
        max_duration: 每个片段处理前 N 秒
        sample_fps: 采样帧率
        output_fps: 输出视频帧率
        use_slicing: 是否使用切片模式
        tile_cols: 横向分块数
        tile_rows: 纵向分块数
        tile_overlap: tile 重叠比例
        split_segments: 是否将完整视频分割为多个片段处理

    Returns:
        统计信息字典（如果是分段模式，返回所有片段的统计信息）
    """
    if api_base_url is None:
        raise ValueError("必须提供 --api_url 参数")

    if prompts is None:
        prompts = TEXT_PROMPTS

    # 标准化提示词：统一翻译为（英文用于API，中文用于显示）
    prompts_for_api, prompts_for_display, en_to_zh_map = _normalize_prompts(prompts)

    # 打开视频获取基本信息
    cap_info = cv2.VideoCapture(video_path)
    if not cap_info.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    src_fps      = cap_info.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / src_fps
    W = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_info.release()

    # 计算分段信息
    if split_segments:
        num_segments = int(total_duration // max_duration) + (1 if total_duration % max_duration > 0 else 0)
        print(f"[INFO] ========== 分段模式 ==========")
        print(f"[INFO] 视频总时长: {total_duration:.1f}s")
        print(f"[INFO] 每段时长: {max_duration}s")
        print(f"[INFO] 分段数量: {num_segments}")
        print(f"[INFO] 分段详情:")
        for i in range(num_segments):
            start = i * max_duration
            end = min((i + 1) * max_duration, total_duration)
            duration = end - start
            last = " (最后一段)" if i == num_segments - 1 else ""
            print(f"[INFO]   片段{i+1}: {start:.1f}s - {end:.1f}s (时长 {duration:.1f}s){last}")
        print(f"[INFO] ===============================\n")

        # 存储所有片段的统计信息
        all_segment_stats = []
        merged_totals = {}

        # 处理每个片段
        for seg_idx in range(num_segments):
            start_time = seg_idx * max_duration
            end_time = min((seg_idx + 1) * max_duration, total_duration)
            segment_duration = end_time - start_time

            segment_id = f"{video_id}_seg{seg_idx+1:02d}"

            print(f"\n{'='*60}")
            print(f"[分段 {seg_idx+1}/{num_segments}] {segment_id}")
            print(f"时间范围: {start_time:.1f}s - {end_time:.1f}s (时长 {segment_duration:.1f}s)")
            print(f"{'='*60}\n")

            # 处理单个片段
            seg_stats = _process_single_segment(
                video_path=video_path,
                segment_id=segment_id,
                output_dir=output_dir,
                api_base_url=api_base_url,
                prompts=prompts,
                prompts_for_api=prompts_for_api,
                prompts_for_display=prompts_for_display,
                en_to_zh_map=en_to_zh_map,
                max_duration=segment_duration,
                start_time=start_time,
                sample_fps=sample_fps,
                output_fps=output_fps,
                use_slicing=use_slicing,
                tile_cols=tile_cols,
                tile_rows=tile_rows,
                tile_overlap=tile_overlap,
            )

            all_segment_stats.append(seg_stats)

            # 合并统计
            for name, count in seg_stats.get("totals", {}).items():
                merged_totals[name] = merged_totals.get(name, 0) + count

        # 创建合并后的统计信息
        merged_stats = {
            "video_id": video_id,
            "source_path": video_path,
            "processed_at": datetime.now().isoformat(timespec='seconds'),
            "resolution": f"{W}x{H}",
            "source_fps": round(src_fps, 2),
            "total_duration_s": round(total_duration, 2),
            "segment_duration_s": max_duration,
            "num_segments": num_segments,
            "text_prompts": {
                "user_input": prompts,
                "display": prompts_for_display,
                "api_calls": prompts_for_api
            },
            "segments": all_segment_stats,
            "merged_totals": merged_totals,
        }

        # 保存合并统计（详细分段信息）
        merged_stats_path = os.path.join(output_dir, f"{video_id}_segments_merged.json")
        with open(merged_stats_path, 'w', encoding='utf-8') as f:
            json.dump(merged_stats, f, ensure_ascii=False, indent=2)

        # 生成符合 server.py 期望格式的统计文件（用于 Web 页面显示）
        # 计算总帧数
        total_frames_processed = sum(seg.get("frames_processed", 0) for seg in all_segment_stats)

        # 使用第一个片段的视频作为示例视频
        first_segment_video = all_segment_stats[0].get("annotated_video", "") if all_segment_stats else ""

        # 生成符合 server.py 期望的格式
        server_stats = {
            "video_id":             video_id,
            "source_path":          video_path,
            "processed_at":         datetime.now().isoformat(timespec='seconds'),
            "resolution":           f"{W}x{H}",
            "source_fps":           round(src_fps, 2),
            "duration_processed_s": round(total_duration, 2),  # 总时长
            "sample_fps":           sample_fps,
            "frames_processed":     total_frames_processed,     # 总帧数
            "annotated_video":      first_segment_video,        # 使用第一个片段的视频
            "api_endpoint":         f"{api_base_url}{API_ENDPOINT}",
            "text_prompts": {
                "user_input": prompts,
                "display": prompts_for_display,
                "api_calls": prompts_for_api
            },
            "thresholds": {
                "box": BOX_THRESHOLD,
                "text": TEXT_THRESHOLD,
                "nms": NMS_THRESHOLD
            },
            "totals":               merged_totals,              # 合并后的统计
            "per_frame":            [],                         # 合并所有片段的逐帧数据
        }

        # 合并所有片段的逐帧数据
        for seg in all_segment_stats:
            server_stats["per_frame"].extend(seg.get("per_frame", []))

        # 保存为 server.py 期望的文件名
        server_stats_path = os.path.join(output_dir, f"{video_id}_stats.json")
        with open(server_stats_path, 'w', encoding='utf-8') as f:
            json.dump(server_stats, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"[分段处理完成] 共处理 {num_segments} 个片段")
        print(f"[合并统计] → {merged_stats_path}")
        print(f"[Web统计] → {server_stats_path}")
        print(f"[总计] {merged_totals}")
        print(f"{'='*60}\n")

        return merged_stats

    # 非分段模式：直接处理整个视频的前 max_duration 秒
    return _process_single_segment(
        video_path=video_path,
        segment_id=video_id,
        output_dir=output_dir,
        api_base_url=api_base_url,
        prompts=prompts,
        prompts_for_api=prompts_for_api,
        prompts_for_display=prompts_for_display,
        en_to_zh_map=en_to_zh_map,
        max_duration=max_duration,
        start_time=0,
        sample_fps=sample_fps,
        output_fps=output_fps,
        use_slicing=use_slicing,
        tile_cols=tile_cols,
        tile_rows=tile_rows,
        tile_overlap=tile_overlap,
    )


def _process_single_segment(video_path, segment_id, output_dir,
                            api_base_url, prompts,
                            prompts_for_api, prompts_for_display, en_to_zh_map,
                            max_duration, start_time,
                            sample_fps, output_fps,
                            use_slicing, tile_cols, tile_rows, tile_overlap):
    """
    处理单个视频片段

    Args:
        video_path: 输入视频路径
        segment_id: 片段 ID
        output_dir: 输出目录
        api_base_url: API 基础 URL
        prompts: 原始提示词（用于 JSON）
        prompts_for_api: 用于 API 调用的英文提示词
        prompts_for_display: 用于显示的中文提示词
        en_to_zh_map: 英文→中文映射表
        max_duration: 处理时长（秒）
        start_time: 起始时间（秒）
        sample_fps: 采样帧率
        output_fps: 输出视频帧率
        use_slicing: 是否使用切片模式
        tile_cols: 横向分块数
        tile_rows: 纵向分块数
        tile_overlap: tile 重叠比例

    Returns:
        统计信息字典
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 计算当前片段的帧范围
    start_frame = int(start_time * src_fps)
    end_frame = min(int((start_time + max_duration) * src_fps), total_frames)
    segment_frame_count = end_frame - start_frame
    sample_interval = max(1, round(src_fps / sample_fps))

    print(f"[INFO] 片段: {segment_id}")
    print(f"[INFO] 分辨率: {W}×{H}  源FPS: {src_fps:.1f}")
    print(f"[INFO] 时间范围: {start_time:.1f}s - {start_time + max_duration:.1f}s")
    print(f"[INFO] 帧范围: {start_frame} - {end_frame} (共 {segment_frame_count} 帧)")
    print(f"[INFO] 采样: 每 {sample_interval} 帧取 1 帧")
    print(f"[INFO] API: {api_base_url}")
    if use_slicing:
        print(f"[INFO] 切片模式: {tile_cols}×{tile_rows} 重叠={int(tile_overlap*100)}%")

    out_video_path = os.path.join(output_dir, f"{segment_id}_annotated.mp4")

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
    sampled      = 0

    print(f"\n[开始处理]")
    first_frame = True

    # 跳帧到起始位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, end_frame):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % sample_interval == 0:
            timestamp = frame_idx / src_fps

            # 调用分割 API（根据配置选择切片或全帧）
            if use_slicing:
                masks = api.segment_frame_with_slicing(
                    frame_bgr, prompts_for_api,
                    BOX_THRESHOLD, TEXT_THRESHOLD, NMS_THRESHOLD,
                    tile_cols, tile_rows, tile_overlap
                )
            else:
                masks = api.segment_frame(
                    frame_bgr, prompts_for_api,
                    BOX_THRESHOLD, TEXT_THRESHOLD, NMS_THRESHOLD
                )

            # 逐帧统计（统一使用中文名称）
            frame_counts = {}
            for mask in masks:
                name_en = mask["name"]
                name_zh = _get_chinese_name_with_map(name_en, en_to_zh_map)
                # 统一使用中文名称统计
                frame_counts[name_zh] = frame_counts.get(name_zh, 0) + 1
                total_counts[name_zh] = total_counts.get(name_zh, 0) + 1

            per_frame.append({
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 2),
                "counts": frame_counts,
                "num_masks": len(masks),
            })

            # 绘制 masks（传入映射表用于中文转换）
            annotated = draw_masks_with_map(frame_bgr, masks, timestamp, en_to_zh_map)

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

    # 刷新剩余帧
    for pkt in av_stream.encode():
        av_output.mux(pkt)
    av_output.close()
    cap.release()

    stats = {
        "video_id":             segment_id,  # 使用 segment_id 作为 video_id，使 server.py 能正确读取片段统计
        "source_path":          video_path,
        "processed_at":         datetime.now().isoformat(timespec='seconds'),
        "resolution":           f"{W}x{H}",
        "source_fps":           round(src_fps, 2),
        "start_time_s":         round(start_time, 2),
        "end_time_s":           round(start_time + max_duration, 2),
        "duration_processed_s": round(max_duration, 2),
        "sample_fps":           sample_fps,
        "frames_processed":     sampled,
        "annotated_video":      f"{segment_id}_annotated.mp4",
        "api_endpoint":         f"{api_base_url}{API_ENDPOINT}",
        "text_prompts": {
            "user_input": prompts,
            "display": prompts_for_display,
            "api_calls": prompts_for_api
        },
        "thresholds": {
            "box": BOX_THRESHOLD,
            "text": TEXT_THRESHOLD,
            "nms": NMS_THRESHOLD
        },
        "totals":               total_counts,
        "per_frame":            per_frame,
    }

    stats_path = os.path.join(output_dir, f"{segment_id}_stats.json")
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
  # 基本用法（处理前N秒）
  python drone_analysis_seg.py --video ./data/test.mp4 --api_url http://localhost:8000
  python drone_analysis_seg.py --video ./data/test.mp4 --api_url http://192.168.1.154:8000 --prompts "icon tower" "utility pole"
  python drone_analysis_seg.py --video ./data/test.mp4 --api_url http://localhost:8000 --duration 5

  # 使用切片模式（SAHI算法）
  python drone_analysis_seg.py --video ./data/test.mp4 --api_url http://localhost:8000 --use_slicing
  python .\\drone_analysis_seg.py --video V:\\datasets\\2026\\4K_video\\DJI_20260125104749_0001_V.MP4 --api_url http://192.168.1.154:8000 --video_id video007 --duration 10 --prompts "铁塔" "光纤" "电线杆" --use_slicing

  # 分段模式（将完整视频按duration分割为多个片段）
  python drone_analysis_seg.py --video ./data/test.mp4 --api_url http://localhost:8000 --duration 10 --split_segments
  python .\\drone_analysis_seg.py --video V:\\datasets\\2026\\4K_video\\DJI_20260125104749_0001_V.MP4 --api_url http://192.168.1.154:8000 --duration 30 --prompts "铁塔" "光纤" "电线杆" --split_segments --use_slicing
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
    # 分段参数
    parser.add_argument("--split_segments", action="store_true", help="启用分段模式：将完整视频按duration分割为多个片段处理")
    # 切片参数
    parser.add_argument("--use_slicing", action="store_true", help="启用切片模式（SAHI算法）")
    parser.add_argument("--tile_cols",  type=int,            help="横向分块数 (默认: 4)", default=TILE_COLS)
    parser.add_argument("--tile_rows",  type=int,            help="纵向分块数 (默认: 2)", default=TILE_ROWS)
    parser.add_argument("--tile_overlap", type=float,        help="tile重叠比例 (默认: 0.1)", default=TILE_OVERLAP)
    args = parser.parse_args()

    # 命令行参数覆盖配置
    use_slicing = args.use_slicing
    split_segments = args.split_segments
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
        split_segments=split_segments,
    )
