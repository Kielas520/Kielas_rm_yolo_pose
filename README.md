# K-Vision

> 面向 RoboMaster 视觉任务的高效端到端关键点检测神经网络框架

本项目是一个用于 **RoboMaster（RM）竞赛** 的装甲板关键点检测系统，覆盖从 ROS2 rosbag 数据提取、清洗均衡、极端数据增强、模型训练，到 ONNX/TorchScript 导出和工业相机实时推理的完整工作流。

---

## 类别映射

系统支持 13 个类别（可通过 `config.yaml` 增删）：

| 编号 | 含义 | Class ID |
|------|------|----------|
| B1 | 蓝方 1 号 装甲板 | 0 |
| B2 | 蓝方 2 号 装甲板 | 1 |
| B3 | 蓝方 3 号 装甲板 | 2 |
| B4 | 蓝方 4 号 装甲板 | 3 |
| B5 | 蓝方 5 号 装甲板 | 4 |
| B7 | 蓝方哨兵 装甲板 | 5 |
| R1 | 红方 1 号 装甲板 | 6 |
| R2 | 红方 2 号 装甲板 | 7 |
| R3 | 红方 3 号 装甲板 | 8 |
| R4 | 红方 4 号 装甲板 | 9 |
| R5 | 红方 5 号 装甲板 | 10 |
| R7 | 红方哨兵 装甲板 | 11 |
| Negative | 负样本（灯管/反光带等） | 12 |

---

## 整体架构

```
ROS2 Bag
   │
   ▼ [tools/extract_ros2_bag.py]
data/raw/{class_id}/{photos, labels}
   │
   ▼ [purify.py]    ← 格式校验、帧间去重、图片缺失检测
data/purify/{class_id}/{photos, labels}
   │
   ▼ [balance.py]   ← 按 max_samples_per_class 下采样、统一标签维度
data/balance/{class_id}/{photos, labels}
   │
   ▼ [split.py]     ← 8:2 划分、补全可见度标志、自动生成 dataset.yaml
data/datasets/{images,labels}/{train,val}
   │
   ▼ [datasets.py + augment.py]  ← CPU 几何增强 + GPU 光度增强
   │
   ▼ [train.py]     ← RMDetector + RMDetLoss + EMA + AdamW + 早停
model_res/best_model.pth
   │
   ▼ [export.py]    ← ONNX + TorchScript 导出
export_res/best_model.onnx
   │
   ▼ [demo.py / detector.py]  ← 多引擎推理 + HIK/USB 相机
实时检测画面
```

---

## 核心特性

### 1. 统一工作流终端

基于 `rich` 构建的交互式控制台（`main.py`），四个核心功能以子进程方式独立运行：

| 选项 | 功能 | 调用的模块 |
|------|------|-----------|
| 1 | 数据预处理全流程 | `src.data_process.process` |
| 2 | 模型训练（支持断点续训） | `src.training.train` |
| 3 | ONNX / TorchScript 导出 | `src.training.export` |
| 4 | 实时推理演示 | `src.demo.demo` |
| 0 | 退出 | — |

Ctrl+C 中断时自动保存最新权重，再按一次强制退出。

---

### 2. 高鲁棒性数据管线

#### 自动维度兼容
训练数据和标签存在两种格式：
- **10 维**：`class_id color/visibility x1 y1 x2 y2 x3 y3 x4 y4`
- **9 维**：`class_id x1 y1 x2 y2 x3 y3 x4 y4`

流水线在 `purify` 阶段可同时读取两种格式；在 `balance` 阶段自动将 10 维降为统一的 9 维；在 `split` 阶段自动补全可见度标志位 `2`，最终在 `datasets.py` 的 `__getitem__` 中兼容 1 维（纯负样本 ID）、9 维、10 维三种输入。

#### 智能清洗（purify.py）
- 按类别目录扫描
- 逐帧校验标签格式（`get_frame_center` 函数）
- **帧间去重**：计算当前帧与上一保存帧的中心点欧氏距离，小于阈值（默认 10px）则跳过
- 统计报告：成功保留数、ID 错误、距离过滤、格式错误、图片缺失、空标签

#### 类别均衡（balance.py）
- 统计每个类别的样本量，超过 `max_samples_per_class`（默认 3000）的类别随机下采样
- 自动计算类别权重：`weight = max_count / count`
- 生成 `train.yaml` 配置文件供训练阶段读取

#### 数据集拆分（split.py）
- 每个类别内部随机打乱，按 `val_ratio`（默认 0.2）划分训练/验证集
- 文件名加 `{class_id}_` 前缀防止跨类别重名
- 生成 `dataset.yaml`，包含类别数、名称映射、类别权重

#### 半在线洗牌机制
训练时每隔 `shuffle_interval`（默认 5）个 epoch，递增 `shared_stage` 共享计数器，DataLoader 的 worker 进程据此重新播种随机数，使同一样本在不同 epoch 看到不同的增强变换，有效防止灾难性遗忘。

---

### 3. 极端数据增强（CPU + GPU 解耦）

增强流水线分为两个阶段，CPU 阶段在 DataLoader worker 内执行几何变换和背景融合，GPU 阶段在 batch 进入模型前批量执行光度变换。

#### CPU 阶段（augment.py:process_cpu）
| 增强 | 说明 | 默认概率 |
|------|------|---------|
| 水平翻转 | 同时翻转关键点顺序 | 0.4 |
| 旋转 | 围绕目标中心旋转（-45° ~ 45°） | 0.6 |
| 缩放 | 根据 `scale_range` 随机缩放目标面积占比 | 0.6 |
| 平移 | 在画面内随机平移，自动裁剪边界 | 0.5 |
| 透视变换 | 随机扭曲四边形视角 | 0.5 |
| 背景替换 | 融合室内背景图片（MIT indoorCVPR_09） | 0.85 |
| 复合背景 | 60% 概率叠加第二张背景 patch | 隐含 |
| 遮挡 | 在关键点附近随机挖矩形洞 | 0.8 |

背景替换时，先计算所有目标的 `expanded_roi`（沿灯条方向外扩），生成 ROI 掩码后高斯模糊，再与背景图 alpha 混合。

#### GPU 阶段（augment.py:process_gpu）
在显存中以 tensor 操作批量执行：

| 增强 | 实现方式 | 默认概率 |
|------|---------|---------|
| 亮度 | `torchvision.adjust_brightness`，增益 0.3~3.0 | 0.8 |
| HSV 偏移 | `adjust_hue` + `adjust_saturation` | 0.4 |
| 高斯模糊 | `gaussian_blur`，核大小 3/5/7/9/11 | 0.7 |
| 高斯噪声 | `torch.randn_like` 叠加，σ=25/255 | 0.7 |
| Bloom 光晕 | 提取 >0.75 高光区域 → 大核模糊 → 叠加回原图 | 0.2 |

---

### 4. 模型架构

#### 整体结构（RMDetector）

```
Backbone (RMBackbone)              Neck (RMNeck: FPN+PAN)          Head (RMHead)
┌────────────────────┐            ┌────────────────────────┐      ┌──────────────────┐
│ Input (3,416,416)  │            │                        │      │                  │
│     │ ConvBNSiLU   │  stride 2  │     Upsample(×2)       │      │  cls_convs(×2)   │
│     ▼              │            │    ┌──────────┐        │      │  ┌────────────┐  │
│ Stage2: 32ch      │  stride 4  │ f5 │ conv1×1  │──▶ P3  │      │  │ ConvBNSiLU │  │
│     ▼              │            │────▶──────────│        │      │  │ ConvBNSiLU │  │
│ Stage3: 64ch ──▶ s3│ stride 8   │    └──────────┘        │      │  └──────┬─────┘  │
│     ▼              │            │    ┌──────────┐        │      │         │        │
│ Stage4: 128ch ──▶ s4│ stride 16 │ f4 │ conv1×1  │──▶ P4  │      │ cls: Conv2d(1×1)│
│     ▼              │            │────▶──────────│        │      │  → num_classes  │
│ Stage5: 256ch ──▶ s5│ stride 32 │    └──────────┘        │      │                  │
│     ▼              │            │  Downsample(stride 2)  │      │  reg_convs(×2)   │
│ SPPF(k=5)          │            │    ┌──────────┐        │      │  ┌────────────┐  │
│ ┌────────────┐     │            │ p3 │ conv3×3  │──▶ P5  │      │  │ ConvBNSiLU │  │
│ │ MaxPool×3  │     │            │────▶──────────│        │      │  │ ConvBNSiLU │  │
│ │ concat     │     │            │    └──────────┘        │      │  └──────┬─────┘  │
│ │ conv1×1    │     │            │  Downsample(stride 2)  │      │         │        │
│ └────────────┘     │            │    ┌──────────┐        │      │ pose: Conv2d(1×1)│
│         256ch ──▶ s5'│          │ p4 │ conv3×3  │──▶ P3  │      │  → 8*reg_max    │
└────────────────────┘            └────────────────────────┘      └──────────────────┘
                                                                       │
                                                            ┌─────────┴─────────┐
                                                            ▼                   ▼
                                                     cls_pred(13ch)     pose_pred(128ch)
                                                     P3: 13×52×52        P3: 128×52×52
                                                     P4: 13×26×26        P4: 128×26×26
                                                     P5: 13×13×13        P5: 128×13×13
```

#### Backbone（RMBackbone）
- **Stage1**：普通 `ConvBNSiLU`（3→16, stride=2）
- **Stage2~5**：`StackedBlocks`（深度可分离卷积堆叠，每阶段 stride=2）
  - 第一块不含残差（负责通道变换 + 下采样）
  - 后续块用残差连接（需 stride=1 且通道数不变）
- **SPPF**：3 个级联的 `MaxPool2d(k=5, stride=1, padding=2)` → concat → `Conv1×1`

#### Neck（RMNeck）
标准 **FPN + PAN** 结构：
- **Top-down**：上采样 → concat 横向连接 → `Conv1×1`
- **Bottom-up**：`Conv3×3 stride=2` 下采样 → concat → `Conv1×1`

#### Head（RMHead）
解耦头设计：
- `cls_convs`（×2 `ConvBNSiLU`）→ `cls_pred`（`Conv2d 1×1`，输出 `num_classes`）
- `reg_convs`（×2 `ConvBNSiLU`）→ `pose_pred`（`Conv2d 1×1`，输出 `8 × reg_max`）

每个尺度输出通道数 = `num_classes + 8 × reg_max`（默认 13 + 128 = 141）。

---

### 5. 损失函数

训练使用五部分复合损失：

```
total_loss = λ_cls × FocalLoss(cls)
           + λ_pose × (
                 DFL(distribution)
               + WingLoss(regression)
               + 0.5 × OKS
               + 0.2 × StructuralLoss
           )
```

#### FocalLoss
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```
- `α = 0.25`（正样本权重），`γ = 2.0`
- 支持 `class_weights`（从 `dataset.yaml` 读取），对少数类施加乘性权重

#### DFL (Distribution Focal Loss)
将关键点坐标建模为 `reg_max`（默认 16）个离散 bin 上的概率分布：
```
DFL = CE(tl) × wl + CE(tr) × wr
```
- `tl = floor(target + reg_max/2)`，`tr = tl + 1`
- `wl = tr - (target + reg_max/2)`，`wr = 1 - wl`

推理时通过 `Integral` 层将分布还原为连续值：
```
pred = Σ(p_i × i) - reg_max/2
```

#### WingLoss
对小误差更敏感的回归损失：
```
Wing(x) = {
    ω × ln(1 + |x|/ε),      if |x| < ω
    |x| - (ω - ω×ln(1+ω/ε)), otherwise
}
```
- `ω = 10`, `ε = 2`

#### OKS (Object Keypoint Similarity)
```
OKS = exp(-d² / (2 × σ² × scale))
```
- `d`：预测与真值的欧氏距离
- `scale = w_bbox × h_bbox`（目标尺度归一化）
- `σ² = 0.05`

#### StructuralLoss
通过对角线中点约束强制四边形结构：
```
Structural = SmoothL1(diag1_mid, center) + SmoothL1(diag2_mid, center)
```
- `diag1_mid = (p0 + p2) / 2`，`diag2_mid = (p1 + p3) / 2`
- `center = mean(p0..p3)`

#### 负样本处理
类别 ID = `negative_class_id`（默认 12）的目标**仅参与分类损失**，不参与任何位姿回归损失。通过 `pos_mask = conf_mask & (class_id != negative_class_id)` 实现分离。

---

### 6. 训练机制

#### 指数移动平均（EMA）
```python
decay = 0.999 × (1 - exp(-updates / 480))
```
- 每步 optimizer.step() 后更新 EMA 权重
- 初期 decay 小，快速跟随；后期趋近 0.999，稳定泛化
- **验证和模型保存均使用 EMA 权重**

#### 学习率调度
- **前 5% epochs 线性预热**：`lr = base_lr × (epoch / warmup_epochs)`
- **余弦退火**：`CosineAnnealingWarmRestarts(T_0=300, T_mult=1, eta_min=1e-6)`

#### 混合精度训练
- `torch.autocast(device_type, dtype=torch.float16)` 自动混合精度
- `GradScaler` 防止梯度下溢
- 梯度裁剪 `max_norm=10.0`

#### 验证指标
```
val_score = w_pck × PCK@5px + w_id × ID_Acc
```
- **PCK@5px**：预测关键点与真值欧氏距离 < 5px 视为正确
- **ID_Acc**：匹配目标中分类正确的比例
- 默认权重：`w_pck = 0.65`, `w_id = 0.35`

PCK 匹配策略：将 GT 与预测按中心距离最小匹配（距离 < `target_in_range_dist`=10px 才算匹配上）。

#### 早停机制
连续 `patience`（默认 60）个 epoch 未创出新高（`val_score`），自动终止训练。

#### 断点续训
`TrainingSessionManager` 上下文管理器确保：
- 正常退出或 Ctrl+C 时自动保存完整 checkpoint
- 原子写入（先写 `.tmp`，再 `replace` 重命名）
- 恢复时加载：模型权重、EMA 状态、优化器状态、Epoch 计数
- 保存训练曲线图和学习率日志

#### 特征可视化钩子
训练结束后自动调用 `visualize_predictions_with_features`，对每个样本：
1. 绘制 GT vs 预测对比图
2. 导出 Neck 层（`conv_f3`, `conv_p4`, `conv_p5`）前 64 通道特征图网格
3. 检测"坏死"通道（全零恒定值）并用红色标记

---

### 7. 推理与部署

#### 多引擎支持

统一的 `Detector` 封装类（`src/demo/src/detector.py`）和 `InferenceEngine`（`src/demo/demo.py`），三种推理后端：

| 后端 | 加载方式 | 优点 |
|------|---------|------|
| PyTorch | `RMDetector + load_state_dict` | 调试方便 |
| ONNX | `onnxruntime.InferenceSession` | 部署轻量、GPU 加速 |
| TorchScript | `torch.jit.load` | C++ 部署兼容 |

#### 关键点解码（decode_tensor）
1. 从特征图读取分类分数（sigmoid 激活）和关键点分布
2. 置信度过滤（`conf_threshold`，默认 0.5~0.7）
3. DFL 还原：softmax → 期望值 → 减 `reg_max/2`
4. 网格坐标解码：`px = (offset_x + grid_x) / grid_w × img_w`
5. 关键点 NMS

#### 基于关键点距离的 NMS（keypoint_nms）

不同于传统 IoU NMS，本系统采用**最小关键点欧氏距离**作为抑制依据：

```
for each prediction pair (i, j):
    dist_matrix = ||pts_i - pts_j||₂  shape: [M, 4, 4]
    min_dist = min(dist_matrix)
    if min_dist < dist_thresh:
        suppress lower-scored prediction
```

- **阈值**：默认 15px（可配置为 `kpt_dist_thresh`）
- **物理意义**：两帧检测共享灯条（角点距离极近）时视为重复，保留高置信度者

#### 相机硬件支持

| 相机类型 | 配置值 | 说明 |
|---------|--------|------|
| USB 摄像头 | `camera_type: "usb"` | OpenCV VideoCapture |
| 海康工业相机 | `camera_type: "hik"` | MVS SDK（MvCameraControl.dll） |

海康相机特性：
- 枚举 USB 和 GigE 设备
- 独占访问、关闭帧率限制
- **热调节曝光**：推理画面下按 `W` 增加曝光、`S` 减少曝光（步长 500μs）
- BayerRG 原始数据自动去马赛克转 RGB

#### 推理绘制
- 跳过负样本类别（不绘制）
- 绘制灯条边缘（两条竖线）+ 四个角点
- 根据类别自动用红/蓝色渲染（cls_id < 6 为红色，否则蓝色）
- 计算水平和垂直方向的外包矩形框
- 左上角显示 FPS 和当前曝光值

---

### 8. 导出

#### ONNX 导出（export.py）
- `torch.onnx.export` 模式，opset 版本 18
- 自动常量折叠（`do_constant_folding=True`）
- 使用 `onnxsim` 进行极致轻量化（常量折叠 + 冗余节点消除）
- 输出三个命名节点：`output_p3`, `output_p4`, `output_p5`

#### TorchScript 导出
- `torch.jit.trace` 跟踪导出
- 保存为 `.pt` 文件

---

## 目录结构

```
K-Vision/
├── config.yaml                     # 全局统一配置
├── main.py                         # 交互式终端入口
├── pyproject.toml                  # UV 项目配置
├── uv.lock                         # UV 锁定文件
├── LICENSE                         # MIT 许可证
├── .gitignore                      # Git 忽略规则
│
├── src/
│   ├── data_process/
│   │   ├── process.py              # 数据流水线控制器
│   │   └── src/
│   │       ├── purify.py           # 数据清洗（去重、格式校验）
│   │       ├── balance.py          # 类别均衡下采样
│   │       ├── split.py            # 训练/验证集划分
│   │       └── visiualize.py       # 采样可视化
│   │
│   ├── training/
│   │   ├── train.py                # 训练主循环（EMA、AMP、早停）
│   │   ├── export.py               # ONNX/TorchScript 导出
│   │   └── src/
│   │       ├── model.py            # RMDetector、decode_tensor、keypoint_nms
│   │       ├── loss.py             # RMDetLoss 复合损失
│   │       ├── datasets.py         # RMArmorDataset + 目标编码
│   │       ├── augment.py          # CPU + GPU 解耦增强
│   │       ├── hook.py             # 特征提取钩子 + 可视化
│   │       └── get_env.py          # CUDA 环境检测
│   │
│   └── demo/
│       ├── demo.py                 # 实时推理主程序
│       └── src/
│           └── detector.py         # Detector 统一封装
│
├── tools/
│   ├── extract_ros2_bag.py         # ROS2 rosbag 数据提取
│   ├── labels.py                   # 交互式负样本标注工具
│   ├── downloader.py               # 背景/负样本数据集下载
│   ├── negative.py                 # 纯负样本批量生成
│   ├── scaler.py                   # 像素尺寸测量辅助
│   └── hik_camera/                 # 海康工业相机 SDK 封装
│       ├── env.py                  # 环境检查
│       ├── main.py                 # 相机独立测试
│       └── src/
│           ├── hik_camera.py       # HikCamera 驱动类
│           └── __init__.py         # DLL 路径配置
│
├── data/                           # 运行时数据目录
│   ├── raw/                        # 原始 rosbag 提取数据
│   ├── purify/                     # 清洗后数据
│   ├── balance/                    # 均衡后数据
│   └── datasets/                   # 最终划分数据集
│       ├── images/{train,val}/
│       └── labels/{train,val}/
│
├── background/                     # 室内背景图片（自动下载）
├── model_res/                      # 训练输出（权重、曲线、日志）
├── export_res/                     # 导出模型存放
└── ros2_bag/                       # ROS2 bag 存放（需手动创建）
```

---

## 快速开始

### 1. 环境配置

```bash
# 安装 uv（如果尚未安装）
pip install uv

# 同步依赖（Windows 自动拉取 cu128 预览版 PyTorch）
uv sync

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. 数据准备

```bash
# 下载室内背景数据集（MIT indoorCVPR_09，约 5000 张）
python tools/downloader.py

# 如果类型需切换为负样本下载，修改 config.yaml:
#   kielas_rm_train.downloader.type: "negative"

# 提取 ROS2 bag（将 bag 放入 ros2_bag/ 目录）
# 文件夹命名映射规则：
#   hero_blue_data → ID 0
#   infantry_blue_data → ID 2
#   sentry_blue_data → ID 5
#   hero_red_data → ID 6
#   infantry_red_data → ID 8
#   sentry_red_data → ID 11
python tools/extract_ros2_bag.py

# 交互式硬负样本采集
# 在 config.yaml 中配置 video_path 和 frame_step
# 播放控制：D 跳过帧 | A 进入标注
# 标注：W/S 切换类别 | 鼠标左键标 4 角点 | 右键撤销 | Enter 保存
python tools/labels.py

# 批量纯负样本生成（从图片目录）
python tools/negative.py
```

### 3. 数据预处理

```bash
python main.py
# 选择 1 → 全流程自动执行：Purify → Visualize → Balance → Visualize → Split → Visualize
```

或分步执行：
```bash
python main.py
# 2: 仅 Purify（清洗）
# 3: 仅 Balance（均衡）
# 4: 仅 Split（拆分）
# 5: Visualize 特定阶段
```

### 4. 模型训练

```bash
python main.py
# 选择 2 → 启动训练

# 如果 model_res/ 目录非空，将提示：
# [1] 继续训练（保留文件，从历史权重恢复）
# [2] 清空并刷新（重新开始）
# [3] 退出任务
```

### 5. 模型导出

```bash
python main.py
# 选择 3 → 导出 ONNX + TorchScript
```

或在 `config.yaml` 的 `kielas_rm_export` 节点配置导出参数后直接运行：
```bash
python -m src.training.export
```

### 6. 实时推理

```bash
python main.py
# 选择 4 → 启动相机实时检测

# 快捷键：
#   W: 增加曝光
#   S: 减少曝光
#   Q: 退出
```

或指定模型和相机参数后直接运行：
```bash
python -m src.demo.demo
```

---

## 配置文件参考

### kielas_rm_train — 训练配置

```yaml
kielas_rm_train:
  downloader:                    # 数据集下载器
    type: "negative"             # "negative" 或 "background"
    background:                  # 背景数据参数
      url: "http://..."          # MIT indoorCVPR_09 下载地址
      limit: 5000                # 最多下载数量
      max_res: 1280              # 最大分辨率过滤
      min_res: 320               # 最小分辨率过滤
    negative:                    # 负样本下载参数
      limit: 5000
      output_dir: "./negative"

  negative_data:                 # 负样本数据路径
    input_dir: "./negative"
    output_dir: "./data/purify"

  sampler:                       # 视频采样器
    video_path: "D:\\Download\\negative.mp4"
    export_dir: "./data/purify"
    frame_step: 5                # 每隔多少帧抽取一次

  dataset:
    raw_dir: "./data/raw"
    purify_dir: "./data/purify"
    balance_dir: "./data/balance"
    datasets_dir: "./data/datasets"
    balance:
      max_samples_per_class: 3000   # 每类最大样本数

    augment:                       # 数据增强参数
      brightness_prob: 0.8
      brightness_range: [0.3, 3.0]
      blur_prob: 0.7
      blur_ksize: [3, 5, 7, 9, 11]
      hsv_prob: 0.4
      hsv_h_gain: 0.01
      hsv_s_gain: 0.3
      hsv_v_gain: 0.3
      noise_prob: 0.7
      bloom_prob: 0.2
      flip_prob: 0.4
      scale_prob: 0.6
      scale_range: [0.01, 0.1]
      rotate_prob: 0.6
      rotate_range: [-45, 45]
      translate_prob: 0.5
      translate_range: 0.6
      perspective_prob: 0.5
      perspective_factor: 0.20
      bg_replace_prob: 0.85
      bg_dir: "./background"
      roi_h_exp: 2.1
      roi_w_exp: 1.1
      occ_prob: 0.8
      occ_radius_pct: 0.4
      occ_size_pct: [0.02, 0.1]

    split:
      val: 0.2                     # 验证集比例

  train:
    device: "cuda"
    num_classes: 13
    negative_class_id: 12
    batch_size: 128
    prefetch_factor: 12
    epochs: 200
    weight_decay: 0.01
    save_dir: "./model_res"
    input_size: [416, 416]
    strides: [8, 16, 32]
    reg_max: 16
    shuffle_interval: 5

    continue:
      path: "./model_res/best_model.pth"

    optimizer:
      base_lr: 0.0002
      betas: [0.940, 0.999]

    ema:
      decay: 0.999
      tau: 480

    scheduler:
      T_0: 300
      T_mult: 1

    post_process:
      conf_threshold: 0.5
      kpt_dist_thresh: 15.0

    data:
      class_id: [0, 2, 5, 6, 8, 11, 12]
      train_img_dir: "./data/datasets/images/train"
      train_label_dir: "./data/datasets/labels/train"
      val_img_dir: "./data/datasets/images/val"
      val_label_dir: "./data/datasets/labels/val"
      num_workers: 16
      scale_ranges: [[0, 140], [120, 250], [237, 640]]

    loss:
      lambda_pose: 5.0
      lambda_cls: 4.0
      alpha: 0.25
      gamma: 2.0
      omega: 10.0
      epsilon: 2.0

    pck:
      target_in_range_dist: 10.0
      max_pixel_threshold: 5.0

    metric_weights:
      pck: 0.65
      id_acc: 0.35

    early_stopping:
      enabled: true
      patience: 60
```

### kielas_rm_export — 导出配置

```yaml
kielas_rm_export:
  weights: "./model_res/best_model.pth"
  num_classes: 13
  negative_class_id: 12
  output_dir: "./export_res"
  formats:
    - "onnx"
    - "torchscript"
  input_size: [416, 416]
  onnx:
    opset: 18
    simplify: true
```

### kielas_rm_demo — 推理配置

```yaml
kielas_rm_demo:
  camera_type: "hik"               # "hik" 或 "usb"
  camera_index: 0
  hik_exposure: 5000               # 海康相机曝光时间（μs）
  hik_gain: 10
  model_type: "onnx"               # "onnx" / "pytorch" / "torchscript"
  model_path: "./export_res/best_model.onnx"
  device: "cuda"
  num_classes: 13
  negative_class_id: 12
  input_size: [416, 416]
  strides: [8, 16, 32]
  conf_threshold: 0.7
  kpt_dist_thresh: 15.0
```

---

## 关键配置项速查表

| 配置路径 | 用途 | 默认值 |
|---------|------|--------|
| `train.input_size` | 模型输入分辨率 | `[416, 416]` |
| `train.strides` | 三个检测尺度的下采样率 | `[8, 16, 32]` |
| `train.reg_max` | DFL 分布 bin 数量 | `16` |
| `train.batch_size` | 训练批大小 | `128` |
| `train.shuffle_interval` | 半在线洗牌间隔 epoch 数 | `5` |
| `train.loss.lambda_pose` | 位姿损失权重 | `5.0` |
| `train.loss.lambda_cls` | 分类损失权重 | `4.0` |
| `demo.conf_threshold` | 推理置信度阈值 | `0.7` |
| `demo.kpt_dist_thresh` | 关键点 NMS 距离阈值 | `15.0` |

---

## 常见问题

**Q：如何添加新的装甲板类别？**
1. 修改 `config.yaml` 中 `train.num_classes` 为新总数
2. 在 `train.data.class_id` 白名单中加入新 ID
3. 在 `data/` 下创建对应 ID 的类别子目录
4. 修改 `tools/extract_ros2_bag.py` 中的 `folder_map` 映射（如果需要从 rosbag 提取）

**Q：如何恢复中断的训练？**
重新运行 `python main.py`，选择 2。当检测到 `model_res/` 非空时选择 `继续训练`。系统会加载 `last_model.pth` 中的完整状态（模型权重 + EMA + 优化器 + Epoch 计数），从断点处继续训练。

**Q：负样本在训练和推理中如何处理？**
- **训练时**：负样本仅参与分类损失，不参与位姿回归（DFL/Wing/OKS/Structural），通过 `negative_class_id` 过滤
- **推理时**：检测结果中类别 ID = `negative_class_id` 的目标跳过绘制

**Q：推理卡顿如何优化？**
- 将 `model_type` 切换为 `"onnx"` 使用 ONNX Runtime 推理
- 确保 `device: "cuda"` 启用 GPU
- 降低输入分辨率（需同步调整 `input_size` 并重新训练）

**Q：海康相机无法打开？**
- 确认 `tools/hik_camera/hik_lib/` 和 `tools/hik_camera/MvImport/` 目录存在（已配置 .gitignore，需手动放置 SDK）
- 运行 `python tools/hik_camera/env.py` 检查 DLL 加载是否成功

**Q：数据增强会不会破坏关键点位置？**
- 几何变换全部使用 `cv2.warpPerspective` 和 `cv2.perspectiveTransform` 进行矩阵精确映射
- 增强结束时有安全性校验（目标主体偏离画面超过 20% 则自动缩放修正）
- 超过 3 个角点出界的标注会被标记为不可见（`vis=0`），在训练时被忽略

---

## 依赖

| 包 | 版本 | 用途 |
|----|------|------|
| Python | ==3.11.* | 运行时 |
| torch | >=2.5.0 | 深度学习框架 |
| torchvision | >=0.19.0 | 图像增强、NMS |
| opencv-python | <4.11 | 图像处理 |
| numpy | <2.0 | 数值计算 |
| onnxruntime(-gpu) | >=1.18,<1.24 | ONNX 推理引擎 |
| onnx | — | ONNX 导出 |
| onnxsim | ==0.4.36 | ONNX 模型简化 |
| onnxscript | — | ONNX 脚本 |
| rich | >=14.3.3 | CLI 交互界面 |
| PyYAML | — | 配置文件读取 |
| matplotlib | >=3.10 | 训练曲线绘制 |
| requests | >=2.33.1 | 数据下载 |
| cvbridge3 | >=1.1 | ROS2 图像桥接 |

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
