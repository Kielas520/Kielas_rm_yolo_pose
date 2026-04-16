# Kielas RM Detector Train

本项目是一个专为 RoboMaster 视觉打造的端到端装甲板关键点检测与分类模型训练框架。
包含从 ROS2 Bag 数据提取、自动化数据清洗与增强，到模型训练、格式导出以及多端推理演示的完整流水线。

## 🌟 核心特性

- **一站式交互终端**：提供基于 `rich` 的命令行 UI (`main.py`)，轻松调度所有流水线脚本。
- **现代化依赖管理**：基于 `uv` 和 `pyproject.toml`，跨平台支持并自动隔离环境。
- **高鲁棒性数据管道**：内置数据去重、类别均衡、定向增强（仅增强训练集）和动态背景替换功能。
- **定制化网络架构**：轻量级 Backbone + PANet 特征融合 + 多分支 Head，联合 Focal Loss、CIoU 与 OKS 损失进行优化。
- **快速部署验证**：一键导出 ONNX/TorchScript，并内置接入普通 USB 相机与海康工业相机的推理 Demo。

## ⚙️ 环境配置

本项目推荐使用 `uv` 进行快速环境构建与包管理。项目根目录已包含 `pyproject.toml`。

```bash
# 1. 安装 uv (如果尚未安装)
pip install uv

# 2. 同步项目依赖 (将根据当前系统自动注入对应的 PyTorch 版本)
uv sync

# 3. 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或者在 Windows 上: .venv\Scripts\activate
```

## 🗂 数据集流转全景

数据集处理严格按照阶段在 `data/` 目录下流转，由 `src/data_process/process.py` (或主菜单) 统一调度：

`raw` -> `purify` -> `balance` -> `datasets` (拆分并执行 `augment`)

### 1. ROS2 Bag 数据采集与提取
录制数据需包含 `/detector/img_debug` 和 `/detector/armors_debug_info` 话题。

支持 6 种主流目标映射配置：
- **0**: 蓝方 1号 (B1) | **6**: 红方 1号 (R1)
- **2**: 蓝方 3号 (B3) | **8**: 红方 3号 (R3)
- **5**: 蓝方 哨兵 (B7) | **11**: 红方 哨兵 (R7)

**操作步骤：**
将录制的包放入 `ros2_bag/` 文件夹中，执行：
```bash
python tools/extract_ros2_bag.py
```
*输出：提取好的原始图片和标签保存在 `data/raw/` 中。*

### 2. 自动化数据处理流水线
通过主入口进入数据处理模块，或直接运行：
```bash
python main.py
# 选择 [1] 数据预处理
```
按顺序将执行以下步骤：
1. **Purify (清洗)**：过滤空标签、解析错误标签，并利用距离阈值剔除冗余高频相似帧。
2. **Balance (均衡)**：对数量过多的类别进行下采样（可在 `config.yaml` 配置上限），并移除标签中的 `color` 字段。
3. **Split (拆分)**：按比例（默认 8:2）拆分 `train` 和 `val` 验证集，生成标准的 `dataset.yaml`，并插入目标可见度标志 `vis=2`。
4. **Augment (增强)**：**仅针对训练集**进行光学、几何、遮挡和背景替换增强，严格保护验证集纯度。

#### 🏷 标签格式演变
- **Raw 阶段 (10列)**: `class_id color x1 y1 x2 y2 x3 y3 x4 y4`
- **训练阶段 (10列)**: `class_id vis x1 y1 x2 y2 x3 y3 x4 y4` 
  *(其中 `vis` 为可见度：0=不可见/严重遮挡, 1=部分遮挡, 2=完全可见)*

## 🚀 模型训练

所有的超参数、数据路径和损失权重均在 `config.yaml` 的 `kielas_rm_train` 节点下配置。

```bash
# 启动训练
python main.py
# 选择 [2] 开启模型训练
```

**训练特性：**
- 动态 DataLoader，规避多进程内存溢出问题。
- 自定义损失评估：**PCK@0.5** (Percentage of Correct Keypoints) 为核心验证指标。
- 支持断点续训与基于 PCK 的自动早停 (Early Stopping)。
- 训练完成后自动在 `model_res/` 下生成 loss 曲线与验证集预测可视化图。

## 📦 格式导出与轻量化

训练获得最佳权重 (`best_model.pth`) 后，可导出用于 C++ 端或部署框架的格式。

```bash
python main.py
# 选择 [3] 模型格式导出
```
支持导出格式（通过 `config.yaml` 配置）：
- **ONNX**: 默认 opset 18，内建 `onnxsim` 极致图优化与算子折叠。
- **TorchScript**: 面向原生 C++ LibTorch 环境。

## 🎥 实时推理演示 (Demo)

验证模型在实际相机流下的识别效果与帧率表现。

```bash
python main.py
# 选择 [4] 实时推理演示
```
**配置 (`config.yaml` -> `kielas_rm_demo`):**
- 支持标准 `usb` 摄像头或 `hik` 海康工业相机。
- 支持在运行时通过热键 `W / S` 动态调整相机曝光度。
- 支持直接加载 `.onnx` 使用 ONNXRuntime 推理，或 `.pt` 加载 TorchScript。

## ⚙️ 配置文件 (config.yaml)

系统行为由项目根目录的 `config.yaml` 全局控制。主要涵盖：
- `kielas_rm_train`: 训练批次、学习率调度、Focal Loss 参数、各类数据增强的概率等。
- `kielas_rm_export`: 导出尺寸、格式和 onnx-simplifier 开关。
- `kielas_rm_demo`: NMS 阈值、置信度阈值、相机设备选择与初始曝光。