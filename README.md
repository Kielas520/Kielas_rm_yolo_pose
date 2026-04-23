# K-Vision

K-Vision 是一个针对 RoboMaster (RM) 视觉任务设计的端到端关键点检测神经网络框架。项目涵盖了从 ROS2 Bag 数据提取、交互式负样本采集、自动化清洗、极端数据增强、模型训练，到最终 ONNX 导出和硬同步推理的完整工作流。

## class_id
| 编号 | 含义             | 序号 |
|------|------------------|------|
| B1   | 蓝方1号 装甲板   | 0    |
| B2   | 蓝方2号 装甲板   | 1    |
| B3   | 蓝方3号 装甲板   | 2    |
| B4   | 蓝方4号 装甲板   | 3    |
| B5   | 蓝方5号 装甲板   | 4    |
| B7   | 蓝方哨兵 装甲板   | 5    |
| R1   | 红方1号 装甲板   | 6    |
| R2   | 红方2号 装甲板   | 7    |
| R3   | 红方3号 装甲板   | 8    |
| R4   | 红方4号 装甲板   | 9    |
| R5   | 红方5号 装甲板   | 10   |
| R7   | 红方哨兵 装甲板   | 11   |
| Negative | 负样本 (灯管/反光带) | 12   |

## 🚀 核心特性

* **统一的控制台入口**: 提供基于 `rich` 构建的 `main.py` 交互式终端，一键调度数据处理、训练、导出与推理演示。
* **极致的数据增强管线**:
    * **CPU 端 (几何与上下文)**: 随机透视变换、旋转、缩放、多边形遮挡（模拟战损与枪管）、全天候背景动态融合。
    * **GPU 端 (光学畸变)**: 基于 `torchvision` 张量运算，极速模拟高斯模糊、HSV 偏移、高增益雪花噪声以及极限光晕 (Bloom)。
* **轻量级且高精度的模型架构**:
    * **Backbone**: 深度可分离卷积 (Depthwise Separable Conv) + SPPF 空间金字塔池化。
    * **Neck & Head**: FPN + PAN 多尺度特征融合，搭配解耦头 (Decoupled Head) 直接回归 8 个角点坐标。
* **复合损失函数**: 结合 Focal Loss (分类)、Distribution Focal Loss (DFL)、Wing Loss (回归)、OKS 以及装甲板中心结构化惩罚损失 (Structural Loss)。
* **部署友好**: 深度适配 ONNX 与 TorchScript，内置基于关键点距离的自定义 NMS，支持标准 UVC 相机与海康 (Hikvision) 工业相机原生调用。

---

## 📂 目录结构

```text
K-Vision/
├── config.yaml                # 全局统一配置文件 (数据、训练、导出、推理)
├── main.py                    # 统一工作流终端控制台入口
├── pyproject.toml             # uv 项目依赖配置 (支持跨平台 Torch 安装)
├── ros2_bag/                  # ROS2 原始录制数据存放目录 (需手动创建)
├── src/                       # 核心源码目录
│   ├── data_process/          # 数据流转: 异常清洗、类别均衡、数据集划分、抽样可视化
│   ├── training/              # 模型训练: 网络构建、复合损失、数据增强、Dataloader
│   ├── demo/                  # 实时推理: Detector 封装、多尺度解码、NMS、相机流接入
│   └── tools/                 # 实用工具链
│       ├── extract_ros2_bag.py# ROS2 rosbag 自动化解析与图片/标签同步提取
│       ├── labels.py          # 交互式视频抽帧标定工具 (用于提取困难负样本/特定目标)
│       ├── background.py      # 室内背景数据集自动下载与分辨率过滤
│       ├── scaler.py          # 像素尺寸辅助测量工具 (调参辅助)
│       └── get_env.py         # CUDA 环境检测脚本
```

---

## 🛠️ 环境配置

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 作为极速包管理器，支持 Python 3.11。项目已配置好跨平台依赖逻辑，在 Windows 下会自动拉取最新的 cu128 预览版 PyTorch。

```bash
# 1. 安装 uv (如果尚未安装)
pip install uv

# 2. 同步并安装所有依赖
uv sync

# 3. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

---

## 🚦 快速开始

所有核心功能均已集成至统一终端菜单中。激活环境后，只需运行：

```bash
python main.py
```

在控制台中，输入对应序号即可执行不同模块的任务：
1. **数据预处理**：自动处理 `data/raw` 下的原始数据（清洗、均衡、拆分）。
2. **开启模型训练**：基于 `config.yaml` 设定的超参数启动训练。
3. **模型格式导出**：将 PyTorch `.pth` 权重导出为部署级 `.onnx`。
4. **实时推理演示**：调用本地或工业相机，加载模型进行实时目标检测。

---

## ⚙️ 工作流指南

### 1. 数据准备 (Data Preparation)

#### ROS2 数据提取
在实车测试时，需要录制包含同步图像与检测信息的 ROS2 Bag 数据。
* **依赖的话题 (Topics)**: 录制时必须包含 `/detector/img_debug` (图像) 和 `/detector/armors_debug_info` (装甲板角点与类别标签)。
* **存放规范**: 将包含 `.db3` 文件的文件夹放入根目录的 `ros2_bag/` 下。文件夹命名必须遵循特定的机器人映射规则，提取脚本会自动将其转换为对应的 ID：
  * `hero_blue_data` -> 0
  * `infantry_blue_data` -> 2
  * `sentry_blue_data` -> 5
  * `hero_red_data` -> 6
  * `infantry_red_data` -> 8
  * `sentry_red_data` -> 11
* **提取指令**: 放置好数据后，运行 `python src/tools/extract_ros2_bag.py`，程序会自动匹配时间戳并解包为图片与 `.txt` 标签至 `data/raw`。

#### 困难负样本采集 (Hard Negatives)
针对赛场中容易引起误识别的高亮日光灯管、场地边缘反光带等，可使用交互式采样工具进行处理。
* 在 `config.yaml` 中配置 `negative_sampler.video_path`，指向录制好的赛场视频。
* 运行采样工具 (如 `src/tools/labels.py`)，程序会按设定的步长抽帧。遇到误识别目标时，按 `A` 键进入标定模式。
* 使用鼠标点击特征角点，并通过键盘 `W/S` 快速将类别切换为 `12` (Negative)。标定完成的数据将作为纯负样本混入训练集，利用 Focal Loss 强制网络学习压制假阳性。

#### 背景图片准备
运行 `src/tools/background.py` 可自动下载并提取 MIT indoorCVPR_09 数据集，用于训练时的背景随机融合增强。

### 2. 模型训练 (Training)
* 训练参数统一在 `config.yaml` 的 `kielas_rm_train` 节点下配置。
* 内置 **半在线洗牌机制** (`shuffle_interval`)，每隔指定 epoch 会刷新多进程数据增强的随机种子。
* 支持 **Forward Hook 特征可视化**：训练或验证结束后，会自动导出 Neck 层 (P3, P4, P5) 的特征图与真实推理对比图，辅助排查模型盲区。

### 3. 模型导出与推理 (Export & Inference)
* **导出模型**: 选择主菜单的 `3` 将模型导出为 ONNX。如果环境中安装了 `onnxsim`，程序会自动对计算图进行极致精简。
* **推理封装**: `src/demo/detector.py` 提供了一个高层级的 `Detector` 类，内部消化了图像缩放、多尺度 Tensor 解码以及基于物理距离的关键点 NMS。
* **海康相机支持**: `demo.py` 无缝对接了底层海康相机接口，可在运行中按下 `W/S` 键实时热更新相机的物理曝光值，应对复杂的赛场打光。

---

## 📝 配置说明 (config.yaml)

所有的超参数调整都在 `config.yaml` 中完成，主要包含四个顶级模块：

* `kielas_rm_train.dataset`: 包含数据清洗的距离阈值、类别均衡最大采样数、各项数据增强概率（如 `bloom_prob`, `occ_prob`、负样本采样步长等）。
* `kielas_rm_train.train`: 训练超参数，包括学习率、优化器配置、早停机制 (`early_stopping`) 以及 PCK 与分类精度的评估权重分配。
* `kielas_rm_export`: 指定导出尺寸、Opset 版本及是否开启化简。
* `kielas_rm_demo`: 硬件相关的配置，如 `camera_type` ("usb" 或 "hik")、海康相机的初始曝光时间与置信度阈值。