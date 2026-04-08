import shutil
import threading
import random
from pathlib import Path
from queue import Queue
from tqdm import tqdm
from collections import defaultdict

def io_worker(task_queue, progress_bar):
    """
    后台 I/O 线程：负责读取旧标签、剔除 color 字段、写入新标签以及拷贝图片
    """
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break
            
        in_label, out_label, in_photo, out_photo = task
        
        # 1. 重新格式化标签：剔除 color (索引为 1 的列)
        with open(in_label, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            # 确保是有效数据行 (至少包含 class_id, color 和 8个角点坐标)
            if len(parts) >= 10: 
                parts.pop(1)  # 移除 color 列
                new_lines.append(" ".join(parts) + "\n")
            
        # 写入剔除了 color 的新格式
        with open(out_label, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        # 2. 拷贝图片文件
        if in_photo and in_photo.exists() and out_photo:
            shutil.copy2(in_photo, out_photo)
            
        progress_bar.update(1)
        task_queue.task_done()

def generate_yaml(output_dir: Path, class_counts: dict, class_weights: dict):
    """
    生成类似 YOLO 的 train.yaml 配置文件，去除 color 定义
    """
    yaml_path = output_dir / "train.yaml"
    sorted_cids = sorted(class_counts.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    yaml_content = f"""# Train Dataset Configuration
path: {output_dir.absolute()}  # 数据集绝对根目录
train: ./  # 训练集相对路径
val: ./    # 验证集相对路径 (视实际划分情况而定)

# 类别数量
nc: {len(class_counts)}

# 类别名称映射
names:
"""
    for cid in sorted_cids:
        yaml_content += f"  {cid}: '{cid}'\n"

    yaml_content += "\n# 类别权重 (用于辅助训练时的 Loss 计算)\nweights:\n"
    for cid in sorted_cids:
        yaml_content += f"  {cid}: {class_weights[cid]:.4f}\n"

    # 移除了 color 的特征说明
    yaml_content += """
# 数据特征格式定义 (Label Format)
# [class_id, l_down_x, l_down_y, l_up_x, l_up_y, r_down_x, r_down_y, r_up_x, r_up_y, center_x, center_y]
features_description:
  - class_id
  - left_light_down_x
  - left_light_down_y
  - left_light_up_x
  - left_light_up_y
  - right_light_down_x
  - right_light_down_y
  - right_light_up_x
  - right_light_up_y
  - center_x
  - center_y
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

def balance_dataset_pipeline(input_dir: str, output_dir: str, max_samples_per_class: int = 3000, num_workers: int = 4):
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.exists():
        print(f"错误：找不到输入目录 {in_path}")
        return

    # 1. 扫描与统计
    class_files = defaultdict(list)
    for class_dir in in_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_id = class_dir.name
        labels_dir = class_dir / "labels"
        photos_dir = class_dir / "photos"

        if not labels_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            photo_file = None
            for ext in ['.jpg', '.png', '.jpeg']:
                temp_photo = photos_dir / (label_file.stem + ext)
                if temp_photo.exists():
                    photo_file = temp_photo
                    break
            class_files[class_id].append((label_file, photo_file))

    if not class_files:
        print("未扫描到有效数据文件。")
        return

    # 2. 下采样与权重计算策略
    selected_files = {}
    for cid, files in class_files.items():
        if len(files) > max_samples_per_class:
            selected_files[cid] = random.sample(files, max_samples_per_class)
        else:
            selected_files[cid] = files

    class_counts = {cid: len(files) for cid, files in selected_files.items()}
    max_count = max(class_counts.values())
    
    # 权重公式
    class_weights = {cid: max_count / count for cid, count in class_counts.items()}
    total_tasks = sum(class_counts.values())

    # 打印平衡策略信息
    print("\n" + "="*45)
    print("数据集平衡策略预览：")
    for cid in sorted(class_counts.keys(), key=lambda x: int(x) if x.isdigit() else x):
        orig_cnt = len(class_files[cid])
        new_cnt = class_counts[cid]
        w = class_weights[cid]
        print(f"类 {cid:<3} | 原数量: {orig_cnt:<6} -> 下采样: {new_cnt:<6} | 权重: {w:.4f}")
    print("="*45 + "\n")

    # 3. 创建输出根目录并生成 train.yaml
    out_path.mkdir(parents=True, exist_ok=True)
    generate_yaml(out_path, class_counts, class_weights)

    # 4. 初始化多线程 Pipeline
    io_queue = Queue(maxsize=1000)
    pbar = tqdm(total=total_tasks, desc="数据转换与转移进度", unit="张")
    
    workers = []
    for _ in range(num_workers):
        t = threading.Thread(target=io_worker, args=(io_queue, pbar), daemon=True)
        t.start()
        workers.append(t)

    # 5. 下发主线程任务
    for cid, files in selected_files.items():
        out_class_dir = out_path / cid
        out_labels_dir = out_class_dir / "labels"
        out_photos_dir = out_class_dir / "photos"
        
        out_labels_dir.mkdir(parents=True, exist_ok=True)
        out_photos_dir.mkdir(parents=True, exist_ok=True)

        for label_file, photo_file in files:
            out_label = out_labels_dir / label_file.name
            out_photo = out_photos_dir / photo_file.name if photo_file else None
            
            io_queue.put((label_file, out_label, photo_file, out_photo))

    # 6. 等待队列及线程完成
    for _ in range(num_workers):
        io_queue.put(None)
        
    for t in workers:
        t.join()

    pbar.close()
    print(f"\n✨ 处理完毕！剔除了 color 的数据集与 train.yaml 已保存至: {out_path.absolute()}")

if __name__ == "__main__":
    balance_dataset_pipeline(
        input_dir="./data/purify", 
        output_dir="./data/balance", 
        max_samples_per_class=3000, 
        num_workers=4
    )