import cv2
import random
import numpy as np
import shutil
import threading
import copy
from pathlib import Path
from queue import Queue
from dataclasses import dataclass
from typing import Tuple
import yaml

# 引入 rich 组件
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TaskProgressColumn, TimeRemainingColumn, ProgressColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

@dataclass
class AugmentConfig:
    """图像增强配置参数类"""
    brightness_prob: float = 0.8
    brightness_range: Tuple[float, float] = (0.6, 1.4)
    flip_prob: float = 0.5
    scale_prob: float = 0.5
    scale_range: Tuple[float, float] = (0.8, 1.2)
    rotate_prob: float = 0.5
    rotate_range: Tuple[float, float] = (-15, 15)
    occ_prob: float = 0.5
    occ_radius_pct: float = 0.2
    occ_size_pct: Tuple[float, float] = (0.05, 0.15)
    vis_heavy_threshold: float = 0.7
    vis_part_threshold: float = 0.1

    @staticmethod
    def from_yaml(yaml_path: str):
        """从 yaml 配置文件加载参数"""
        cfg = AugmentConfig()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                # 定位到 config.yaml 中的 augment 层级
                aug_data = data.get('kielas_rm_model', {}).get('dataset', {}).get('augment', {})
                
                if aug_data:
                    # 遍历 yaml 中的键值对，如果匹配则更新
                    for key, value in aug_data.items():
                        if hasattr(cfg, key):
                            # 处理元组类型 (yaml 读入通常是 list)
                            if isinstance(value, list) and len(value) == 2:
                                value = tuple(value)
                            setattr(cfg, key, value)
                    console.print(f"[green]已从 {yaml_path} 加载增强配置[/green]")
                else:
                    console.print("[yellow]YAML 中未发现 augment 配置，使用默认参数[/yellow]")
        except Exception as e:
            console.print(f"[red]加载 YAML 失败: {e}，将使用默认参数[/red]")
        return cfg

class MofNCompleteColumn(ProgressColumn):
    """自定义列显示 n/m 格式"""
    def render(self, task):
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        return Text(f"{completed}/{total}", style="progress.remaining")


def parse_labels(label_lines, filename=""):
    """解析标签，向下兼容 9个值 和 10个值"""
    parsed = []
    for line in label_lines:
        clean_line = line.replace(',', ' ').strip()
        parts = clean_line.split()
        if not parts or len(parts) < 9:
            continue
        class_id = parts[0]
        try:
            if len(parts) == 9:
                visibility = 2
                pts = np.array([float(x) for x in parts[1:9]]).reshape(-1, 2)
            else:
                visibility = int(float(parts[1]))
                pts = np.array([float(x) for x in parts[2:10]]).reshape(-1, 2)
            parsed.append({'class_id': class_id, 'vis': visibility, 'pts': pts})
        except ValueError:
            continue
    return parsed


def format_labels(labels):
    """将结构化标签转回字符串列表"""
    new_lines = []
    for lab in labels:
        pts_flat = lab['pts'].flatten()
        coords_str = " ".join([f"{coord:.6f}" for coord in pts_flat])
        new_lines.append(f"{lab['class_id']} {lab['vis']} {coords_str}")
    return new_lines


def process_data(img, labels, cfg: AugmentConfig):
    """同步处理图像与标签增强"""
    aug_img = img.copy()
    aug_labels = copy.deepcopy(labels)
    h, w = aug_img.shape[:2]
        
    # 1. 亮度调整
    if random.random() < cfg.brightness_prob:
        factor = random.uniform(*cfg.brightness_range)
        aug_img = np.clip(aug_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    
    # 1.5 水平翻转 (针对装甲板四角点排序优化)
    if random.random() < cfg.flip_prob:
        aug_img = cv2.flip(aug_img, 1)  # 1 表示水平翻转
        for lab in aug_labels:
            # 1. 翻转所有坐标点的 x 值：新 x = 图像宽度 - 原 x
            lab['pts'][:, 0] = w - lab['pts'][:, 0]
            
            # 2. 重新排序点位以符合定义:
            # 原顺序: [0:左下, 1:左上, 2:右下, 3:右上]
            # 翻转后: 原左下变为右下，原左上变为右上，以此类推
            # 因此需要执行交换：(左下 <-> 右下), (左上 <-> 右上)
            
            # 暂存翻转后的点数据
            old_pts = lab['pts'].copy()
            
            # 执行索引交换
            lab['pts'][0] = old_pts[2]  # 新左下 = 旧右下
            lab['pts'][1] = old_pts[3]  # 新左上 = 旧右上
            lab['pts'][2] = old_pts[0]  # 新右下 = 旧左下
            lab['pts'][3] = old_pts[1]  # 新右上 = 旧左上

    # 2. 随机 Resize 
    if random.random() < cfg.scale_prob:
        scale = random.uniform(*cfg.scale_range)
        aug_img = cv2.resize(aug_img, None, fx=scale, fy=scale)
        for lab in aug_labels:
            lab['pts'] = lab['pts'] * scale
        h, w = aug_img.shape[:2] # 更新当前尺寸
            
    # 3. 旋转处理
    if random.random() < cfg.rotate_prob:
        angle = random.uniform(*cfg.rotate_range)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_img = cv2.warpAffine(aug_img, M, (w, h))
        
        theta = np.radians(-angle) 
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        for lab in aug_labels:
            pts = lab['pts'] - np.array(center)
            rotated_pts = np.empty_like(pts)
            rotated_pts[:, 0] = pts[:, 0] * cos_t - pts[:, 1] * sin_t
            rotated_pts[:, 1] = pts[:, 0] * sin_t + pts[:, 1] * cos_t
            lab['pts'] = rotated_pts + np.array(center)

    # 4. 围绕目标的随机遮挡 (Cutout)
    if aug_labels and random.random() < cfg.occ_prob:
        radius = (w * cfg.occ_radius_pct) / 2.0
        occ_boxes = []
        
        for lab in aug_labels:
            if random.random() < 0.5:
                tx, ty = np.mean(lab['pts'], axis=0)
                angle = random.uniform(0, 2 * np.pi)
                dist = random.uniform(0, radius)
                cx, cy = tx + dist * np.cos(angle), ty + dist * np.sin(angle)
                
                occ_w = int(w * random.uniform(*cfg.occ_size_pct))
                occ_h = int(h * random.uniform(*cfg.occ_size_pct))
                
                occ_x1, occ_y1 = int(cx - occ_w / 2), int(cy - occ_h / 2)
                occ_x2, occ_y2 = occ_x1 + occ_w, occ_y1 + occ_h
                occ_boxes.append((occ_x1, occ_y1, occ_x2, occ_y2))
                
                draw_y1, draw_y2 = max(0, occ_y1), min(h, occ_y2)
                draw_x1, draw_x2 = max(0, occ_x1), min(w, occ_x2)
                if draw_y1 < draw_y2 and draw_x1 < draw_x2:
                    aug_img[draw_y1:draw_y2, draw_x1:draw_x2] = 0
        
        if occ_boxes:
            for lab in aug_labels:
                # 基于点的遮挡辅助判断
                covered_points = 0
                for pt in lab['pts']:
                    px, py = pt[0], pt[1]
                    if any(x1 <= px <= x2 and y1 <= py <= y2 for x1, y1, x2, y2 in occ_boxes):
                        covered_points += 1

                # 基于面积的遮挡判断
                pts = lab['pts']
                min_x, min_y = np.min(pts, axis=0)
                max_x, max_y = np.max(pts, axis=0)
                target_area = (max_x - min_x) * (max_y - min_y)
                
                is_heavily_occluded = False
                is_partially_occluded = False

                if target_area > 0:
                    for x1, y1, x2, y2 in occ_boxes:
                        ix1, iy1 = max(min_x, x1), max(min_y, y1)
                        ix2, iy2 = min(max_x, x2), min(max_y, y2)
                        if ix1 < ix2 and iy1 < iy2:
                            overlap_ratio = ((ix2 - ix1) * (iy2 - iy1)) / target_area
                            if overlap_ratio > cfg.vis_heavy_threshold:
                                is_heavily_occluded = True
                            elif overlap_ratio > cfg.vis_part_threshold:
                                is_partially_occluded = True

                if covered_points == 4 or is_heavily_occluded:
                    lab['vis'] = 0
                elif covered_points > 0 or is_partially_occluded:
                    lab['vis'] = min(lab['vis'], 1)
        
    return aug_img, aug_labels


def augment_worker(task_queue: Queue, progress: Progress, task_id, cfg: AugmentConfig):
    """后台线程：执行增强处理任务"""
    while True:
        task = task_queue.get()
        if task is None: break
        img_path, out_img_path, label_path, out_label_path = task
        try:
            parsed_labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    parsed_labels = parse_labels(f.readlines(), filename=label_path.name)
            
            img = cv2.imread(str(img_path))
            if img is None:
                progress.advance(task_id)
                task_queue.task_done()
                continue
                
            aug_img, aug_labels = process_data(img, parsed_labels, cfg)
            cv2.imwrite(str(out_img_path), aug_img)
            
            new_label_lines = format_labels(aug_labels)
            with open(out_label_path, 'w') as f:
                if new_label_lines:
                    f.write("\n".join(new_label_lines) + "\n")
        except Exception as e:
            progress.console.print(f"[red]处理异常 {img_path.name}: {e}[/red]")
        progress.advance(task_id)
        task_queue.task_done()


def generate_yaml(output_dir: Path):
    """生成 train.yaml 配置文件"""
    yaml_path = output_dir / "train.yaml"
    class_counts = {}
    for class_dir in output_dir.iterdir():
        if class_dir.is_dir() and (class_dir / "labels").exists():
            count = len(list((class_dir / "labels").glob("*.txt")))
            if count > 0: class_counts[class_dir.name] = count

    if not class_counts: return
    max_count = max(class_counts.values())
    class_weights = {cid: max_count / count for cid, count in class_counts.items()}
    sorted_cids = sorted(class_counts.keys(), key=lambda x: int(x) if x.isdigit() else x)

    content = f"path: {output_dir.absolute()}\ntrain: ./\nval: ./\nnc: {len(class_counts)}\n\nnames:\n"
    for cid in sorted_cids: content += f"  {cid}: '{cid}'\n"
    content += "\nweights:\n"
    for cid in sorted_cids: content += f"  {cid}: {class_weights[cid]:.4f}\n"
    with open(yaml_path, 'w', encoding='utf-8') as f: f.write(content)


def run_augment_pipeline(input_dir: str, output_dir: str, num_workers: int = 8, cfg: AugmentConfig = None):
    if cfg is None: cfg = AugmentConfig()
    in_path, out_path = Path(input_dir), Path(output_dir)

    if not in_path.exists():
        console.print(f"[bold red]错误：[/bold red]找不到输入目录 {in_path}")
        return

    tasks = []
    with console.status("[bold green]正在扫描原始数据..."):
        for class_dir in in_path.iterdir():
            if not class_dir.is_dir(): continue
            photos_dir, labels_dir = class_dir / "photos", class_dir / "labels"
            if not photos_dir.exists(): continue
            for img_file in photos_dir.glob("*.jpg"):
                label_file = labels_dir / (img_file.stem + ".txt")
                out_class_dir = out_path / class_dir.name
                tasks.append((img_file, out_class_dir / "photos" / f"aug_{img_file.name}", 
                              label_file, out_class_dir / "labels" / f"aug_{label_file.name}"))

    if not tasks: return
    if out_path.exists(): shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # 打印配置信息
    table = Table(title="数据增强流水线", header_style="bold cyan")
    table.add_column("配置项")
    table.add_column("当前值")
    table.add_row("输入/输出", f"{input_dir} -> {output_dir}")
    table.add_row("并行线程数", str(num_workers))
    table.add_row("亮度范围", f"{cfg.brightness_range} (概率:{cfg.brightness_prob})")
    table.add_row("水平翻转概率", f"{cfg.flip_prob}") 
    table.add_row("旋转角度", f"{cfg.rotate_range}")
    table.add_row("遮挡半径占比", f"{cfg.occ_radius_pct}")
    console.print(table)

    progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                        BarColumn(bar_width=40), TaskProgressColumn(), MofNCompleteColumn(),
                        TimeRemainingColumn(), console=console)

    with progress:
        main_task = progress.add_task("[magenta]执行增强处理...", total=len(tasks))
        task_queue = Queue(maxsize=2000)
        threads = [threading.Thread(target=augment_worker, args=(task_queue, progress, main_task, cfg), daemon=True) 
                   for _ in range(num_workers)]
        for t in threads: t.start()

        created_dirs = set()
        for t in tasks:
            if t[1].parent not in created_dirs:
                t[1].parent.mkdir(parents=True, exist_ok=True)
                t[3].parent.mkdir(parents=True, exist_ok=True)
                created_dirs.add(t[1].parent)
            task_queue.put(t)

        for _ in range(num_workers): task_queue.put(None)
        for t in threads: t.join()

    generate_yaml(out_path)
    console.print(Panel(f"✅ 处理完成！总数: {len(tasks)}", border_style="green", title="Success"))

if __name__ == "__main__":
    # 1. 从 YAML 加载配置
    config_path = "config.yaml"
    config = AugmentConfig.from_yaml(config_path)

    # 2. 执行流水线
    run_augment_pipeline(
        input_dir="./data/balance", 
        output_dir="./data/augment", 
        num_workers=8,
        cfg=config
    )