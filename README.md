# K-Vision

> 面向 RoboMaster 视觉任务的端到端关键点检测神经网络框架

K-Vision 是一个针对 **RoboMaster（RM）竞赛** 设计的高效、端到端装甲板关键点检测框架，覆盖从 ROS2 数据提取、清洗均衡、极端数据增强、模型训练，到 ONNX/TorchScript 导出和工业相机实时推理的全流程。

---

## 总览

```
rosbag ──▶ extract ──▶ purify ──▶ balance ──▶ split ──▶ augment ──▶ train ──▶ export ──▶ deploy
  │                   │           │            │          │            │         │           │
  │   sensor_msgs/    │  剔除     │  按类别    │  8:2     │ CPU+GPU    │ EMA     │ ONNX/     │ USB/
  │   Image +         │  无效/    │  下采样    │  划分    │ 极端增强   │ 早停    │ Torch-    │ HIK
  │   armors_debug    │  去重帧   │  防过拟合  │          │            │         │ Script    │ 相机
```

---

## 类别映射（默认 13 类）

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

> 类别可在 `config.yaml` 中灵活增删。

---

## 核心特性

### 统一工作流终端
基于 `rich` 构建的交互式控制台（`main.py`），一键调度：
1. 数据预处理（清洗/均衡/拆分/可视化）
2. 模型训练（多进程混合增强）
3. 模型导出（ONNX + TorchScript）
4. 实时推理演示（USB / 海康工业相机）

### 高鲁棒性数据管线
- **自动维度兼容**：自动兼容 10 维（含颜色/可见度）与 9 维（纯坐标）标签，底层统一处理
- **智能清洗**：基于帧间中心点距离的去重、格式校验、图片缺失检测
- **类别均衡**：按 `max_samples_per_class` 上限下采样，防止过拟合；自动生成类别权重供训练使用
- **半在线洗牌**：每隔固定 Epoch 刷新多进程随机种子，打破数据序列聚集

### 极端数据增强（CPU + GPU 解耦）

| 阶段 | 增强操作 | 执行位置 |
|------|----------|----------|
| 几何变换 | 翻转、旋转、缩放、平移、透视变换 | CPU（DataLoader worker） |
| 背景融合 | 室内背景替换、复合背景拼接 | CPU |
| 遮挡 | 角点附近随机矩形遮挡 | CPU |
| 光度变换 | 亮度调整 | GPU（batch） |
| 色彩变换 | HSV 偏移 | GPU |
| 模糊 | 高斯模糊（可变核大小） | GPU |
| 噪声 | 高斯噪声 | GPU |
| 光晕 | Bloom 效果（高光模糊泛光） | GPU |

### 轻量级高精度模型架构

```
Backbone                          Neck (FPN+PAN)            Head
┌─────────────┐                  ┌──────────────────┐      ┌──────────────┐
│ ConvBNSiLU    │   stride 2     │      up          │      │ cls_convs     │──▶ cls_pred [num_classes]
│ StackedBlocks │──▶ feat_s3 ──▶│   ┌──────┐        │      │              │
│ StackedBlocks │──▶ feat_s4 ──▶│──▶│ conv │──▶ P3   │      │ reg_convs    │──▶ pose_pred [8×reg_max]
│ StackedBlocks │──▶ feat_s5 ──▶│    └──────┘        │      └──────────────┘
│ SPPF          │               │   down             │
└─────────────┘                  │   ┌──────┐        │
                                 │──▶│ conv │──▶ P5   │
                                 │    └──────┘        │
                                 │   down             │
                                 │   ┌──────┐        │
                                 │──▶│ conv │──▶ P4   │
                                 │    └──────┘        │
                                 └──────────────────┘
```

- **Backbone**：深度可分离卷积块（`DepthwiseConvBlock`）+ 残差连接 + SPPF（快速空间金字塔池化）
- **Neck**：标准 FPN + PAN 结构，输出 P3/P4/P5 三个尺度
- **Head**：解耦头，分类和回归分支独立卷积
- **输出**：每个尺度输出 `num_classes + 8×reg_max` 个通道

### 复合损失函数
```python
loss = λ_cls × FocalLoss(cls) + λ_pose × (
    DFL(distribution) +
    WingLoss(regression) +
    0.5 × OKS(
        1 - exp(-d² / (2 × σ² × scale))
    ) +
    0.2 × StructuralLoss(
        SmoothL1(diag_midpoint, center)
    )
)
```

| 损失 | 作用 |
|------|------|
| Focal Loss | 分类分支，缓解正负样本不均衡 |
| Distribution Focal Loss (DFL) | 关键点位置的概率分布回归 |
| Wing Loss | 关键点绝对坐标回归，对小误差敏感度可调 |
| OKS | 基于目标尺度归一化的关键点相似度 |
| Structural Loss | 对角线中点 → 中心点，强制四边形结构一致性 |

### 部署友好
- **ONNX 导出**：自动常量折叠 + onnxsim 极致轻量化
- **TorchScript 导出**：JIT trace 导出
- **多引擎推理**：统一 `Detector` 封装，支持 ONNX Runtime / PyTorch / TorchScript
- **物理距离感知 NMS**：基于关键点最小欧氏距离（而非 IoU）的 NMS
- **硬同步相机**：原生支持海康工业相机 MVS SDK；推理画面下 W/S 键实时调节曝光

---

## 训练机制

### 指数移动平均（EMA）
```python
ema_decay = 0.999 × (1 - exp(-updates / 480))
```
- 训练初期 decay 较小，快速跟随模型
- 后期趋于 0.999，保持稳定泛化能力
- 验证和保存均使用 EMA 权重

### 半在线洗牌
每隔 `shuffle_interval`（默认 5）个 Epoch，刷新所有 DataLoader worker 的随机种子（`shared_stage` 值递增），使同一样本在不同 Epoch 看到不同的增强变换。

### 综合评估分数
```
val_score = 0.65 × PCK@5px + 0.35 × ID_Acc
```
- **PCK**：预测关键点与真值距离 < 5px 视为正确
- **ID_Acc**：匹配目标中分类正确的比例

### 早停机制
连续 `patience`（默认 60）个 Epoch 未创新高，自动终止训练。

---

## 目录结构

```
K-Vision/
├── config.yaml                 # 全局统一配置文件
├── main.py                     # 统一工作流终端控制台入口
├── pyproject.toml              # uv 项目依赖配置
├── src/
│   ├── data_process/           # 数据流水线
│   │   └── src/
│   │       ├── purify.py       # 数据清洗（去重、格式校验）
│   │       ├── balance.py      # 类别均衡下采样
│   │       ├── split.py        # 训练/验证集划分
│   │       └── visiualize.py   # 样本可视化
│   ├── training/               # 训练模块
│   │   ├── train.py            # 训练主循环
│   │   ├── export.py           # ONNX/TorchScript 导出
│   │   └── src/
│   │       ├── model.py        # 网络结构（Backbone/Neck/Head）
│   │       ├── loss.py         # 复合损失函数
│   │       ├── datasets.py     # 数据集与目标编码
│   │       ├── augment.py      # CPU + GPU 双阶段数据增强
│   │       └── hook.py         # 特征图可视化钩子
│   ├── demo/
│   │   ├── demo.py             # 实时推理主程序
│   │   └── src/
│   │       └── detector.py     # Detector 推理封装
├── tools/
│   ├── extract_ros2_bag.py     # ROS2 bag 提取
│   ├── labels.py               # 负样本交互式采集
│   ├── background.py           # 背景数据集下载
│   ├── scaler.py               # 像素尺寸测量
│   └── hik_camera/             # 海康工业相机 SDK
└── export_res/                 # 导出模型存放
```

---

## 快速开始

### 1. 环境配置
```bash
pip install uv
uv sync
.venv\Scripts\activate    # Windows
source .venv/bin/activate  # Linux/macOS
```

### 2. 数据准备
```bash
# 下载背景数据集
python src/tools/background.py

# 提取 ROS2 bag 数据（将 bag 放入 ros2_bag/）
python src/tools/extract_ros2_bag.py

# 交互式负样本采集
python src/tools/labels.py
```

### 3. 数据预处理
```bash
python main.py
# 选择 1 → 自动执行: Purify → Balance → Split → Visualize
```

### 4. 模型训练
```bash
python main.py
# 选择 2 → 启动训练（支持断点续训）
```

### 5. 模型导出
```bash
python main.py
# 选择 3 → 导出 ONNX + TorchScript
```

### 6. 实时推理
```bash
python main.py
# 选择 4 → 启动相机实时检测（W/S 调曝光，Q 退出）
```

---

## 配置文件参考

```yaml
# ==================== 数据增强 ====================
kielas_rm_train:
  dataset:
    augment:
      brightness_prob: 0.8          # 亮度调整概率
      brightness_range: [0.3, 3.0]  # 亮度增益范围
      blur_prob: 0.7                 # 模糊概率
      hsv_prob: 0.4                  # HSV 偏移概率
      noise_prob: 0.7                # 高斯噪声概率
      bloom_prob: 0.2                # 光晕概率
      flip_prob: 0.4                 # 水平翻转概率
      scale_prob: 0.6                # 随机缩放概率
      rotate_prob: 0.6               # 随机旋转概率（±45°）
      translate_prob: 0.5            # 随机平移概率
      perspective_prob: 0.5          # 透视变换概率
      bg_replace_prob: 0.85          # 背景替换概率
      occ_prob: 0.8                  # 遮挡概率

# ==================== 训练超参数 ====================
  train:
    num_classes: 13                  # 类别数
    batch_size: 128                  # 批大小
    epochs: 200                      # 最大训练轮数
    input_size: [416, 416]           # 输入尺寸
    strides: [8, 16, 32]             # 下采样步长
    reg_max: 16                      # DFL 分布范围
    shuffle_interval: 5              # 半在线洗牌间隔
    optimizer:
      base_lr: 0.0002                # 基础学习率
      betas: [0.940, 0.999]
    ema:
      decay: 0.999                   # EMA 衰减率
      tau: 480                       # 预热步数
    scheduler:
      T_0: 300                       # 余弦退火周期
    loss:
      lambda_pose: 5.0               # 位姿损失权重
      lambda_cls: 4.0                # 分类损失权重
    early_stopping:
      enabled: true
      patience: 60                   # 早停耐心值

# ==================== 推理配置 ====================
kielas_rm_demo:
  camera_type: "hik"                 # usb 或 hik
  model_type: "onnx"                 # onnx / pytorch / torchscript
  conf_threshold: 0.7               # 置信度阈值
  kpt_dist_thresh: 15.0             # 关键点 NMS 距离阈值
```

---

## 常见问题

**Q：如何添加新的类别？**
在 `config.yaml` 中修改 `num_classes`，确保数据目录包含对应 ID 的子文件夹，并在 `class_id` 白名单中加入新 ID。

**Q：如何继续中断的训练？**
重新运行 `python main.py` 选择 2，当检测到 `model_res/` 非空时选择 `继续训练`，自动加载 `last_model.pth` 的完整状态（权重、优化器、学习率、Epoch 计数）。

**Q：推理时如何更换相机？**
修改 `config.yaml` 中 `kielas_rm_demo.camera_type` 为 `"usb"` 或 `"hik"`，并调整 `hik_exposure` / `hik_gain`。

---

## License

MIT
