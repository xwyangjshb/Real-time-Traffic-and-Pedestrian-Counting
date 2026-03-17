# Real-time-Traffic-and-Pedestrian-Counting

# Introduction
This project focuses "counting and statistics of moving targets we care about", driven by YOLOv3 implemented in TensorFlow 2.x.

It needs to be stated that the YOLOv3 detector of this project is forked from the nice implementation of [YunYang1994](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3)

# Project Demo

- The demo is available on Youtube and Bilibili
- on my laptop gtx1060 FPS reached 12-20

![car](https://github.com/Clemente420/Real-time-Traffic-and-Pedestrian-Counting/blob/master/docs/car.png "car")
![person](https://github.com/Clemente420/Real-time-Traffic-and-Pedestrian-Counting/blob/master/docs/person.png "person")

---

# Installation

## 方式一：原始环境（CPU only）

适用于原始交通计数功能，TensorFlow 2.0 + Python 3.6：

```bash
conda env create -f environment.yml
conda activate <env_name>
```

下载 YOLOv3 官方权重并放到项目根目录：

```bash
# Linux / macOS
wget https://pjreddie.com/media/files/yolov3.weights

# Windows 请手动下载后放置到项目根目录
# https://pjreddie.com/media/files/yolov3.weights
```

Two test videos are prepared [here](https://drive.google.com/drive/folders/16ZYObAm48Y0ImnCjtUIzeasyp2QaPphI?usp=sharing).

---

## 方式二：GPU 加速环境（推荐，用于无人机视频分析）

适用于 `drone_analysis.py`，支持 NVIDIA GPU 推理加速。

**环境要求：**
- Windows 10/11
- NVIDIA 显卡，驱动版本支持 CUDA 11.x（`nvidia-smi` 查看）
- Miniconda 或 Anaconda

**第一步：创建 conda 环境**

```bash
conda create -n tf_gpu python=3.9 -y
conda activate tf_gpu
```

**第二步：安装 CUDA 工具库**（conda 自动处理所有 DLL 依赖，无需手动安装 CUDA Toolkit）

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
```

**第三步：安装 TensorFlow 2.10**（最后一个支持 Windows 原生 GPU 的版本）

```bash
pip install tensorflow==2.10.0
pip install numpy==1.23.5   # 固定版本避免兼容性问题
```

**第四步：安装项目其余依赖**

```bash
pip install opencv-python filterpy scipy flask av easydict Pillow
```

**第五步：验证 GPU 是否识别**

```bash
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

预期输出包含：
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

> ⚠️ **注意：** TensorFlow >= 2.11 在 Windows 原生环境下**不支持 GPU**，请务必使用 TF 2.10，或改用 WSL2 运行。

---

# Parameter adjustment

- For `video_demo.py`:
  - `video_path = "./vehicle.mp4"`
  - `num_classes = 80`
  - `utils.load_weights(model, "./yolov3.weights")`
- For `core/utils.py`:
  - `specified_class_id_filter = 2`（2=car, 0=person）
  - `line = [(0, 530), (2100, 530)]`（计数线坐标）

- For `drone_analysis.py`（顶部配置区）:
  - `MAX_DURATION = 10`（处理前 N 秒）
  - `SAMPLE_FPS = 5`（抽帧率）
  - `TARGET_CLASSES`（目标类别 ID，当前为 COCO 占位符）
  - `LAN_TARGET_IP`（Flask 服务所在内网 IP，用于 OPNsense NAT 脚本）

---

# Run

**原始交通计数：**

```bash
conda activate <env_name>
python video_demo.py
```

**无人机视频离线分析（drone_analysis.py）：**

```bash
conda activate tf_gpu
cd Real-time-Traffic-and-Pedestrian-Counting

# 处理视频，生成标注视频和统计 JSON
python drone_analysis.py --video <视频文件路径> --video_id <自定义ID>
# 示例：
python drone_analysis.py --video DJI_h264_cpu.mp4 --video_id demo_001

# 启动 Web 查询服务
python server.py
# 访问 http://localhost:8080
```

---

# Citation

If you use this code for your publications, please cite it as:

```
@ONLINE{vdtc,
    author = "Clemente420",
    title  = "Real-time-Traffic-and-Pedestrian-Counting",
    year   = "2020",
    url    = "https://github.com/Clemente420/Real-time-Traffic-and-Pedestrian-Counting"
}
```

# Author

- Please contact for dataset or more info: clemente0620@gmail.com

# License

This system is available under the MIT license. See the LICENSE file for more info.
