import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

# ---------------------------------------------------------
# 1. 目标编码逻辑 (上一回合推导的算法)
# ---------------------------------------------------------
def encode_target(label_data, img_w=320, img_h=320, grid_w=10, grid_h=10):
    """
    将原始标签转化为 13维的特征图网格张量
    label_data: [class_id, visibility, x1, y1, x2, y2, x3, y3, x4, y4]
    """
    kpts = np.array(label_data[2:]).reshape(4, 2)
    
    x_min, y_min = np.min(kpts, axis=0)
    x_max, y_max = np.max(kpts, axis=0)
    
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    w, h = x_max - x_min, y_max - y_min
    
    cx_norm, cy_norm = cx / img_w, cy / img_h
    w_norm, h_norm = w / img_w, h / img_h
    kpts_norm = kpts / np.array([img_w, img_h])
    
    g_x = int(np.clip(cx_norm * grid_w, 0, grid_w - 1))
    g_y = int(np.clip(cy_norm * grid_h, 0, grid_h - 1))
    
    t_x = cx_norm * grid_w - g_x
    t_y = cy_norm * grid_h - g_y
    
    kpts_grid_offset = kpts_norm * np.array([grid_w, grid_h]) - np.array([g_x, g_y])
    kpts_offset_flat = kpts_grid_offset.flatten()
    
    target_vector = np.zeros(13, dtype=np.float32)
    target_vector[0] = 1.0 
    target_vector[1:5] = [t_x, t_y, w_norm, h_norm]
    target_vector[5:13] = kpts_offset_flat
    
    class_id = int(label_data[0])
    
    return target_vector, g_x, g_y, class_id


# ---------------------------------------------------------
# 2. PyTorch Dataset 封装
# ---------------------------------------------------------
class RMArmorDataset(Dataset):
    def __init__(self, img_dir, label_dir, input_size=(320, 320), grid_size=(10, 10), transform=None):
        """
        img_dir: 图像文件夹路径 (例如: images/train)
        label_dir: 标签文件夹路径 (例如: labels/train)
        input_size: 网络输入尺寸 (W, H)
        grid_size: 特征图网格尺寸 (W_grid, H_grid)
        transform: 数据增强操作 (可选)
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.grid_size = grid_size
        self.transform = transform
        
        # 获取所有有效的数据样本文件名
        self.samples = [f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.txt')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        
        # 1. 读取图像
        img_path = os.path.join(self.img_dir, f"{sample_name}.jpg") # 假设格式为jpg，根据实际情况调整
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV 默认 BGR，转为 RGB
        orig_h, orig_w = img.shape[:2]
        
        # 2. 读取标签
        label_path = os.path.join(self.label_dir, f"{sample_name}.txt")
        with open(label_path, 'r') as f:
            # 假设每行一个目标，这里以单目标为例。如果是多目标需要遍历解析
            line = f.readline().strip()
            # 你的标签格式可能包含前缀如 "0 2..."，需要做一下字符串切割处理
            # 提取出纯数字部分: [class_id, visibility, x1, y1...]
            parts = line.split(']')[-1].strip().split() 
            label_data = [float(x) for x in parts]

        # 3. 图像缩放 (Resize) 与坐标等比例映射
        # 注意：这里需要将原始图像的坐标转换到 320x320 下的坐标
        scale_x = self.input_size[0] / orig_w
        scale_y = self.input_size[1] / orig_h
        img_resized = cv2.resize(img, self.input_size)
        
        # 缩放坐标数据 (跳过前两位的 class_id 和 visibility)
        for i in range(2, len(label_data)):
            if i % 2 == 0: # X 坐标
                label_data[i] *= scale_x
            else:          # Y 坐标
                label_data[i] *= scale_y

        # 4. 如果有数据增强，在这里执行 (如 Mosaic, 颜色抖动等)
        if self.transform:
            # 增强逻辑需同步更新 img_resized 和 label_data 的坐标
            pass 

        # 5. 初始化空白的目标 Tensor (13, H_grid, W_grid) 和 分类标签 Tensor (1, H_grid, W_grid)
        target_tensor = np.zeros((13, self.grid_size[1], self.grid_size[0]), dtype=np.float32)
        class_tensor = np.zeros((1, self.grid_size[1], self.grid_size[0]), dtype=np.int64)

        # 6. 调用编码函数并填入 Tensor
        target_vec, g_x, g_y, class_id = encode_target(
            label_data, 
            img_w=self.input_size[0], img_h=self.input_size[1], 
            grid_w=self.grid_size[0], grid_h=self.grid_size[1]
        )
        
        target_tensor[:, g_y, g_x] = target_vec
        class_tensor[0, g_y, g_x] = class_id

        # 7. 将图像转为 PyTorch 需要的 Shape: (C, H, W) 并归一化到 [0, 1]
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        
        return img_tensor, torch.from_numpy(target_tensor), torch.from_numpy(class_tensor)