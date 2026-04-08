import random
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_dataset(root_path: str, data_type: str = "train", if_color = True):
    # 1. 初始化路径对象
    root = Path(root_path)
    type_dir = root / data_type
    
    if not type_dir.exists():
        print(f"错误: 找不到目录 {type_dir}")
        return

    # 2. 动态分析分类文件夹 (文件夹名即为 Class ID)
    # 过滤出所有子目录
    class_dirs = sorted([d for d in type_dir.iterdir() if d.is_dir()])
    num_classes = len(class_dirs)
    
    if num_classes == 0:
        print(f"在 {type_dir} 下未发现分类文件夹")
        return

    # 3. 创建画布: 行数为类别数，列数为 3 (每个类抽3张)
    fig, axes = plt.subplots(num_classes, 3, figsize=(15, 4 * num_classes))
    
    # 处理只有一个类别的特殊绘图情况
    if num_classes == 1:
        axes = [axes]

    for row, class_dir in enumerate(class_dirs):
        class_id = class_dir.name
        photo_dir = class_dir / "photos"
        label_dir = class_dir / "labels"
        
        # 获取所有图片并随机抽样
        all_photos = list(photo_dir.glob("*.jpg"))
        if not all_photos:
            print(f"类别 {class_id} 中没有图片")
            continue
            
        selected_photos = random.sample(all_photos, min(len(all_photos), 3))
        
        for col in range(3):
            ax = axes[row][col]
            if col >= len(selected_photos):
                ax.axis('off')
                continue
                
            img_path = selected_photos[col]
            # 寻找同名的 .txt 标签
            label_path = label_dir / img_path.with_suffix('.txt').name
            
            # 读取图片 (OpenCV 不直接支持 Path 对象，需转为 str)
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 读取并绘制标签内容
            if label_path.exists():
                with label_path.open('r') as f:
                    for line in f:
                        # 解析示例数据: 0 501.0 605.0 499.0 560.0 689.0 595.0 685.0 553.0
                        parts = list(map(float, line.split()))
                        if len(parts) < 9:
                            continue
                        if if_color == True:
                            # 坐标映射
                            # pts 格式: [左下, 左上, 右下, 右上]
                            pts = [
                                (int(parts[1+1]), int(parts[2+1])), # ld
                                (int(parts[3+1]), int(parts[4+1])), # lu
                                (int(parts[5+1]), int(parts[6+1])), # rd
                                (int(parts[7+1]), int(parts[8+1]))  # ru
                            ]
                        else:
                            # 坐标映射
                            # pts 格式: [左下, 左上, 右下, 右上]
                            pts = [
                                (int(parts[1]), int(parts[2])), # ld
                                (int(parts[3]), int(parts[4])), # lu
                                (int(parts[5]), int(parts[6])), # rd
                                (int(parts[7]), int(parts[8]))  # ru
                            ]
                        
                        # 绘制左灯条 (lu -> ld) 和 右灯条 (ru -> rd)
                        cv2.line(img, pts[1], pts[0], (0, 255, 0), 2)
                        cv2.line(img, pts[3], pts[2], (0, 255, 0), 2)
                        
                        # 绘制四个顶点
                        for p in pts:
                            cv2.circle(img, p, 4, (255, 0, 0), -1)

            ax.imshow(img)
            ax.set_title(f"Class: {class_id} | {img_path.name}", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# 调用示例
visualize_dataset("./data", "raw", True)