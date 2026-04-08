# Kielas_rm_detector_train
手搓适合 rm 视觉体质的 yolo pose 模型来识别装甲板
## 数据集采样
- 录制数据
共计 12 个目标类别，具体映射如下：

| **类别序号** | **目标对象** | **类别序号** | **目标对象** |
| --- | --- | --- | --- |
| **0** | 蓝方 1号 (B1) or hero_blue_data | **6** | 红方 1号 (R1) hero_blue_data |
| **2** | 蓝方 3号 (B3) or infantry_blue_data | **8** | 红方 3号 (R3) or infantry_red_data |
| **5** | 蓝方 哨兵 (B7) or sentry_blue_data | **11** | 红方 哨兵 (R7) or sentry_red_data |
```bash
    ros2 bag record -o infantry_red_data /image_raw /detector/armors_debug_info
```
- 把6份录制的数据文件夹放在 **extract_ros2_bag/original** 文件夹中
- 使用 [extract.py](extract_ros2_bag/extract.py) 对包进行采样
    - 处理好的文件会按照 classid 存放在 **data/raw** 文件夹中
    - labels 会保存在 **data/labels** 文件夹中
    ```
        # label.txt
        class_id :0 - 11, 
        color: 0 - 1 (red or blue), 
        left_light_down: x, left_light_down: y, 
        left_light_up: x, left_light_up: y, 
        right_light_down: x, right_light_down: y, 
        right_light_up: x, right_light_up: y, 
        center: x, center: y
    ```
## 数据集净化
- 使用 [purify.py](src/data_process/purify.py) 对数据进行净化
- 净化后的数据会保存在 **data/purified** 文件夹中
## 数据平衡
- 使用 [balance.py](src/data_process/balance.py) 对数据进行平衡
## 数据增强
- 使用 [augment.py](src/data_process/augment.py) 对数据进行增强
- 数据增强会顺便对数据进行平权和构建完整的数据集
- 构建好的数据集会保存在 **data/augmented** 文件夹中
## 训练模型
- 使用 **data/augmented** 数据集
- 使用 [train.py](train.py) 对数据进行训练