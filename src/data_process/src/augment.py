import cv2
import random
import numpy as np
import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class AugmentConfig:
    # --- 光学增强 ---
    brightness_prob: float = 0.9
    brightness_range: Tuple[float, float] = (0.2, 3.5)
    blur_prob: float = 0.7
    blur_ksize: List[int] = None # type: ignore
    hsv_prob: float = 0.8
    hsv_h_gain: float = 0.010
    hsv_s_gain: float = 0.7
    hsv_v_gain: float = 0.8
    noise_prob: float = 0.7
    bloom_prob: float = 0.8

    # --- 几何形变 ---
    flip_prob: float = 0.5
    scale_prob: float = 0.9
    # 此处定义为目标面积占图像总面积的比例范围
    scale_area_ratio: Tuple[float, float] = (0.01, 0.8) 
    rotate_prob: float = 0.8
    rotate_range: Tuple[float, float] = (-45, 45)
    translate_prob: float = 0.8
    translate_range: float = 0.4
    perspective_prob: float = 0.8
    perspective_factor: float = 0.35

    # --- 背景与遮挡 ---
    bg_replace_prob: float = 0.85
    bg_dir: str = "./background"
    occ_prob: float = 0.8
    occ_radius_pct: float = 0.4
    occ_size_pct: Tuple[float, float] = (0.02, 0.35)
    
    def __post_init__(self):
        if self.blur_ksize is None:
            self.blur_ksize = [3, 5, 7, 9, 11]

def generate_composite_bg(bg_paths, w, h):
    """生成带有堆叠效果的复合背景"""
    if not bg_paths:
        return np.zeros((h, w, 3), dtype=np.uint8)
        
    bg = cv2.imread(str(random.choice(bg_paths)))
    if bg is None: return np.zeros((h, w, 3), dtype=np.uint8)
    bg = cv2.resize(bg, (w, h))

    # 概率堆叠 1-2 个子背景块
    if random.random() < 0.6:
        for _ in range(random.randint(1, 2)):
            patch = cv2.imread(str(random.choice(bg_paths)))
            if patch is None: continue
            pw, ph = random.randint(int(w*0.3), int(w*0.7)), random.randint(int(h*0.3), int(h*0.7))
            patch = cv2.resize(patch, (pw, ph))
            px, py = random.randint(0, w - pw), random.randint(0, h - ph)
            bg[py:py+ph, px:px+pw] = patch
            
    return bg

def process_data(img, labels, cfg: AugmentConfig, bg_paths: str = ''): 
    aug_img = img.copy()
    aug_labels = copy.deepcopy(labels)
    h_orig, w_orig = aug_img.shape[:2]
    
    # --- 0. 基础 HSV 调整 (在融合前做，避免影响新背景) ---
    if random.random() < cfg.hsv_prob:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_gain = random.uniform(-cfg.hsv_h_gain, cfg.hsv_h_gain) * 180
        s_gain = random.uniform(-cfg.hsv_s_gain, cfg.hsv_s_gain)
        v_gain = random.uniform(-cfg.hsv_v_gain, cfg.hsv_v_gain)
        hsv[:, :, 0] = (hsv[:, :, 0] + h_gain) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + s_gain), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + v_gain), 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < cfg.brightness_prob:
        factor = random.uniform(*cfg.brightness_range)
        aug_img = np.clip(aug_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # --- 1. 创建目标蒙版 ---
    mask = np.zeros((h_orig, w_orig), dtype=np.float32)
    if aug_labels:
        for lab in aug_labels:
            pts = lab['pts'].astype(np.int32)
            # 点位顺序: [左下(0), 左上(1), 右下(2), 右上(3)]
            # 转换为顺时针多边形连点顺序: 0 -> 1 -> 3 -> 2
            hull_pts = np.array([pts[0], pts[1], pts[3], pts[2]])
            cv2.fillPoly(mask, [hull_pts], 1.0)
            
        # 膨胀一点点蒙版，带上周围真实光晕的过渡边缘
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    # --- 2. 核心几何变换 ---
    if aug_labels:
        # 水平翻转
        if random.random() < cfg.flip_prob:
            aug_img = cv2.flip(aug_img, 1)
            mask = cv2.flip(mask, 1)
            for lab in aug_labels:
                lab['pts'][:, 0] = w_orig - lab['pts'][:, 0]
                old_pts = lab['pts'].copy()
                lab['pts'][0] = old_pts[2] 
                lab['pts'][1] = old_pts[3] 
                lab['pts'][2] = old_pts[0] 
                lab['pts'][3] = old_pts[1] 

        # 选取一个主装甲板作为变换锚点
        primary_pts = aug_labels[0]['pts']
        hull_pts = np.array([primary_pts[0], primary_pts[1], primary_pts[3], primary_pts[2]], dtype=np.float32)
        plate_area = max(cv2.contourArea(hull_pts), 1.0)
        cx, cy = np.mean(primary_pts, axis=0)

        # -- 计算缩放系数 --
        scale = 1.0
        if random.random() < cfg.scale_prob:
            target_ratio = random.uniform(*cfg.scale_area_ratio)
            target_area = w_orig * h_orig * target_ratio
            scale = np.sqrt(target_area / plate_area)
            
            # 限制边界：防止缩放后装甲板本身就大于蒙版
            min_x, min_y = np.min(primary_pts, axis=0)
            max_x, max_y = np.max(primary_pts, axis=0)
            plate_w, plate_h = max_x - min_x, max_y - min_y
            max_scale_w = w_orig / (plate_w + 1e-6)
            max_scale_h = h_orig / (plate_h + 1e-6)
            scale = min(scale, max_scale_w * 0.95, max_scale_h * 0.95)
        
        angle = random.uniform(*cfg.rotate_range) if random.random() < cfg.rotate_prob else 0.0

        # -- 以锚点为中心构建旋转缩放矩阵 --
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        R = cv2.getRotationMatrix2D((0, 0), angle, scale)
        R = np.vstack([R, [0, 0, 1]])
        
        # -- 计算平移与边界约束 --
        pts_centered = primary_pts - [cx, cy]
        pts_rotated = cv2.transform(np.array([pts_centered], dtype=np.float32), R[:2, :])[0]
        min_tx, min_ty = np.min(pts_rotated, axis=0)
        max_tx, max_ty = np.max(pts_rotated, axis=0)

        valid_cx_min, valid_cx_max = -min_tx, w_orig - max_tx
        valid_cy_min, valid_cy_max = -min_ty, h_orig - max_ty

        if random.random() < cfg.translate_prob:
            ncx = random.uniform(max(0, valid_cx_min), min(w_orig, valid_cx_max))
            ncy = random.uniform(max(0, valid_cy_min), min(h_orig, valid_cy_max))
        else:
            # 即使不强制平移，也要确保变换后的目标没有飞出视野
            ncx = np.clip(cx, valid_cx_min, valid_cx_max)
            ncy = np.clip(cy, valid_cy_min, valid_cy_max)

        T2 = np.array([[1, 0, ncx], [0, 1, ncy], [0, 0, 1]])
        M_affine = (T2 @ R @ T1)[:2, :]

        # 统一执行仿射变换 (图像 + 蒙版 + 点位)
        aug_img = cv2.warpAffine(aug_img, M_affine, (w_orig, h_orig), borderValue=(0, 0, 0))
        mask = cv2.warpAffine(mask, M_affine, (w_orig, h_orig), flags=cv2.INTER_NEAREST, borderValue=0)
        for lab in aug_labels:
            lab['pts'] = cv2.transform(np.array([lab['pts']], dtype=np.float32), M_affine)[0]

        # -- 透视畸变 --
        if random.random() < cfg.perspective_prob:
            margin = min(h_orig, w_orig) * cfg.perspective_factor
            pts1 = np.float32([[0, 0], [w_orig, 0], [0, h_orig], [w_orig, h_orig]]) # type: ignore
            pts2 = np.float32([ # type: ignore
                [random.uniform(0, margin), random.uniform(0, margin)],
                [w_orig - random.uniform(0, margin), random.uniform(0, margin)],
                [random.uniform(0, margin), h_orig - random.uniform(0, margin)],
                [w_orig - random.uniform(0, margin), h_orig - random.uniform(0, margin)]
            ])
            M_persp = cv2.getPerspectiveTransform(pts1, pts2) # type: ignore
            aug_img = cv2.warpPerspective(aug_img, M_persp, (w_orig, h_orig), borderValue=(0, 0, 0))
            mask = cv2.warpPerspective(mask, M_persp, (w_orig, h_orig), flags=cv2.INTER_NEAREST, borderValue=0)
            for lab in aug_labels:
                lab['pts'] = cv2.perspectiveTransform(np.array([lab['pts']], dtype=np.float32), M_persp)[0]

    # --- 3. 关键点遮挡生成 (在 Mask 上挖洞) ---
    if aug_labels and random.random() < cfg.occ_prob:
        radius = min(w_orig, h_orig) * cfg.occ_radius_pct
        for lab in aug_labels:
            for pt in lab['pts']:
                if random.random() < 0.5:
                    angle = random.uniform(0, 2 * np.pi)
                    dist = random.uniform(0, radius)
                    cx, cy = pt[0] + dist * np.cos(angle), pt[1] + dist * np.sin(angle)
                    
                    occ_w = int(w_orig * random.uniform(*cfg.occ_size_pct))
                    occ_h = int(h_orig * random.uniform(*cfg.occ_size_pct))
                    
                    # 模拟枪管的极度细长条或大面积黑块遮挡
                    if random.random() < 0.5: 
                        occ_w, occ_h = occ_h, occ_w 

                    x1, y1 = int(cx - occ_w/2), int(cy - occ_h/2)
                    x2, y2 = x1 + occ_w, y1 + occ_h
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    # --- 4. 背景替换与组合 ---
    if bg_paths and random.random() < cfg.bg_replace_prob:
        bg_img = generate_composite_bg(bg_paths, w_orig, h_orig)
        mask_blur = cv2.GaussianBlur(mask, (7, 7), 0)
        mask_3d = np.expand_dims(mask_blur, axis=-1)
        # ROI内的板子保留，其余黑边和被遮挡挖出的洞全由背景填充
        aug_img = (aug_img.astype(np.float32) * mask_3d + bg_img.astype(np.float32) * (1 - mask_3d)).astype(np.uint8)

    # --- 5. 最终画质劣化 (全局统一) ---
    if random.random() < cfg.blur_prob:
        ksize = random.choice(cfg.blur_ksize)
        aug_img = cv2.blur(aug_img, (ksize, ksize))

    if random.random() < cfg.noise_prob:
        noise = np.random.normal(0, 20, aug_img.shape).astype(np.float32)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if aug_labels and random.random() < cfg.bloom_prob:
        bloom_layer = np.zeros_like(aug_img, dtype=np.float32)
        for lab in aug_labels:
            center = np.mean(lab['pts'], axis=0).astype(int)
            cv2.circle(bloom_layer, tuple(center), int(min(h_orig, w_orig) * 0.05), (255, 255, 255), -1)
        bloom_layer = cv2.GaussianBlur(bloom_layer, (31, 31), sigmaX=20)
        aug_img = np.clip(aug_img.astype(np.float32) + bloom_layer * 0.4, 0, 255).astype(np.uint8)

    # 最后边界清理
    for lab in aug_labels:
        out_count = sum(1 for pt in lab['pts'] if pt[0] < 0 or pt[0] >= w_orig or pt[1] < 0 or pt[1] >= h_orig)
        if out_count >= 3:  
            lab['vis'] = 0

    return aug_img, aug_labels

# ================= 测试代码 =================
if __name__ == "__main__":
    def parse_labels_for_test(label_path):
        parsed = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    visibility = int(parts[1]) if len(parts) > 9 else 2
                    offset = 2 if len(parts) > 9 else 1
                    pts = np.array([float(x) for x in parts[offset:offset+8]]).reshape(-1, 2)
                    parsed.append({'class_id': parts[0], 'vis': visibility, 'pts': pts})
        return parsed
        
    cfg = AugmentConfig()
    
    dataset_dir = Path("./data/datasets")
    train_images = list((dataset_dir / "images" / "train").glob("*.jpg"))
    train_labels_dir = dataset_dir / "labels" / "train"
    
    # 抽取3张图片测试
    test_samples = random.sample(train_images, min(3, len(train_images))) if train_images else []
    
    bg_dir = Path(cfg.bg_dir)
    bg_paths = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png")) if bg_dir.exists() else []
    
    out_dir = Path("./data/test/augment")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"提取了 {len(test_samples)} 张图片准备测试...")
    
    for i, img_path in enumerate(test_samples):
        img = cv2.imread(str(img_path))
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        
        labels = []
        if label_path.exists():
            labels = parse_labels_for_test(label_path)
            
        # 生成10个变体观察效果
        for v in range(10):
            aug_img, aug_lbls = process_data(img, labels, cfg, bg_paths)
            
            # 画上绿点用于检查几何映射对不对
            viz_img = aug_img.copy()
            for lbl in aug_lbls:
                if lbl['vis'] > 0:
                    for pt in lbl['pts']:
                        cv2.circle(viz_img, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                        
            out_path = out_dir / f"test_{img_path.stem}_v{v}.jpg"
            cv2.imwrite(str(out_path), viz_img)
            
    print(f"测试完毕，输出文件已保存至 {out_dir}")