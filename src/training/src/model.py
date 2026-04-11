import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    """标准的 3x3 卷积块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseConvBlock(nn.Module):
    """深度可分离卷积块 (Depthwise Separable Convolution)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x


class RMBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBNReLU(3, 16, stride=2)
        self.stage2 = DepthwiseConvBlock(16, 32, stride=2)
        self.stage3 = DepthwiseConvBlock(32, 64, stride=2)
        self.stage4 = DepthwiseConvBlock(64, 128, stride=2)
        self.stage5 = DepthwiseConvBlock(128, 256, stride=2)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        feat_stage3 = self.stage3(x)
        x = self.stage4(feat_stage3)
        feat_stage5 = self.stage5(x)
        return feat_stage3, feat_stage5
    
class RMNeck(nn.Module):
    """特征融合层 (FPN 变体)"""
    def __init__(self, in_channels_s3=64, in_channels_s5=256, out_channels=256):
        super().__init__()
        self.downsample_s3 = nn.Sequential(
            nn.Conv2d(in_channels_s3, in_channels_s3, kernel_size=4, stride=4, 
                      groups=in_channels_s3, bias=False),
            nn.BatchNorm2d(in_channels_s3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_s3, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.fuse = nn.Sequential(
            ConvBNReLU(64 + in_channels_s5, out_channels),
            DepthwiseConvBlock(out_channels, out_channels)
        )

    def forward(self, feat_s3, feat_s5):
        feat_s3_down = self.downsample_s3(feat_s3)
        out = torch.cat([feat_s3_down, feat_s5], dim=1) 
        out = self.fuse(out) 
        return out


class RMHead(nn.Module):
    """解耦检测头 (Decoupled Head)"""
    def __init__(self, in_channels=256):
        super().__init__()
        self.box_head = nn.Conv2d(in_channels, 5, kernel_size=1, stride=1)
        self.pose_head = nn.Conv2d(in_channels, 8, kernel_size=1, stride=1)

    def forward(self, x):
        box_out = self.box_head(x)   
        pose_out = self.pose_head(x) 
        out = torch.cat([box_out, pose_out], dim=1)
        return out


class RMDetector(nn.Module):
    """完整的单阶段装甲板检测模型"""
    def __init__(self):
        super().__init__()
        self.backbone = RMBackbone()
        self.neck = RMNeck(in_channels_s3=64, in_channels_s5=256, out_channels=256)
        self.head = RMHead(in_channels=256)

    def forward(self, x):
        feat_s3, feat_s5 = self.backbone(x)
        fused_feat = self.neck(feat_s3, feat_s5)
        out = self.head(fused_feat)
        return out


# ==========================================
# 新增：核心解码工具函数 (供推断与可视化使用)
# ==========================================
def decode_tensor(tensor, is_pred=True, conf_threshold=0.5, grid_size=(13, 13), img_size=(416, 416)):
    """
    将 13 维的网格张量解码为真实的物理像素坐标
    可以同时处理真实标签 (GT) 和网络预测 (Pred)
    """
    batch_size = tensor.shape[0]
    grid_w, grid_h = grid_size
    img_w, img_h = img_size
    
    # 如果是预测结果，置信度通道需要经过 Sigmoid 激活；真实标签则不需要
    if is_pred:
        conf = torch.sigmoid(tensor[:, 0, :, :])
    else:
        conf = tensor[:, 0, :, :]
        
    batch_results = []
    
    for b in range(batch_size):
        # 1. 过滤低于置信度阈值的网格
        mask = conf[b] >= conf_threshold
        if not mask.any():
            batch_results.append([])
            continue
            
        # 2. 拿到有目标的网格索引
        grid_y, grid_x = torch.nonzero(mask, as_tuple=True)
        scores = conf[b, grid_y, grid_x]
        
        # 3. 提取位姿数据 (8个偏移量)
        # 【修复点】：PyTorch 高级索引取出的形状是 [8, N]，需要转置为 [N, 8]
        raw_pose = tensor[b, 5:13, grid_y, grid_x].T  
        decoded_pose = torch.zeros_like(raw_pose)
        
        # 4. 反向解码 4 点坐标
        for i in range(4): 
            px_offset = raw_pose[:, i*2]
            py_offset = raw_pose[:, i*2 + 1]
            
            # 计算全图归一化坐标 [0, 1]
            px_norm = (px_offset + grid_x) / grid_w
            py_norm = (py_offset + grid_y) / grid_h
            
            # 转换为实际像素坐标
            decoded_pose[:, i*2] = px_norm * img_w
            decoded_pose[:, i*2 + 1] = py_norm * img_h
            
        # 拼合结果: [score, x1, y1, x2, y2, x3, y3, x4, y4]
        dets = torch.cat([scores.unsqueeze(1), decoded_pose], dim=1)
        batch_results.append(dets.detach().cpu().numpy())
        
    return batch_results

# 测试完整模型
if __name__ == "__main__":
    model = RMBackbone()
    dummy_input = torch.randn(1, 3, 320, 320)
    out3, out5 = model(dummy_input)
    print(f"Stage 3 Output Shape: {out3.shape}") # 预期: [1, 64, 40, 40]
    print(f"Stage 5 Output Shape: {out5.shape}") # 预期: [1, 256, 10, 10]
    # 实例化模型
    model = RMDetector()
    
    # 模拟输入：Batch Size 为 2，3 通道，320x320
    dummy_input = torch.randn(2, 3, 320, 320)
    
    # 前向传播
    output = model(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}") 
    # 预期输出形状: torch.Size([2, 13, 10, 10])，与 dataset 中生成的 target_tensor 完全对应
    