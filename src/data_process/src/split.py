import random
import shutil
import threading
from pathlib import Path
from queue import Queue
import yaml

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, ProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

class MofNCompleteColumn(ProgressColumn):
    def render(self, task):
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        return Text(f"{completed}/{total}", style="progress.remaining")

def format_and_copy_label(src_path: Path, dst_path: Path):
    """读取标签，如果是 9 个值，则在第二列补全可见度标志 2，然后写入新路径"""
    try:
        with open(src_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.replace(',', ' ').strip().split()
            if not parts: continue
            
            # 如果是原始的 9 值（类别 + 8个坐标），插入默认可见度 2
            if len(parts) == 9:
                parts.insert(1, '2')
                
            new_lines.append(" ".join(parts))
            
        with open(dst_path, 'w', encoding='utf-8') as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")
    except Exception as e:
        raise RuntimeError(f"标签格式化失败: {e}")

def io_worker(task_queue: Queue, progress: Progress, task_id):
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break
            
        src_photo, dst_photo, src_label, dst_label = task
        
        try:
            # 图片保持原样拷贝
            if src_photo and src_photo.exists():
                shutil.copy2(src_photo, dst_photo)
            
            # 标签文件进行格式化后写入
            if src_label and src_label.exists():
                format_and_copy_label(src_label, dst_label)
        except Exception as e:
            console.print(f"[red]文件处理错误 {src_photo.name}: {e}[/red]")
            
        progress.advance(task_id)
        task_queue.task_done()

def generate_yaml(input_dir: Path, output_dir: Path):
    class_counts = {}
    for class_dir in input_dir.iterdir():
        if class_dir.is_dir() and (class_dir / "labels").exists():
            count = len(list((class_dir / "labels").glob("*.txt")))
            if count > 0: class_counts[class_dir.name] = count

    if not class_counts:
        return

    nc = len(class_counts)
    sorted_cids = sorted(class_counts.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    
    max_count = max(class_counts.values())
    weights_dict = {cid: max_count / count for cid, count in class_counts.items()}

    yaml_content = f"""# Target Detection Dataset Configuration
path: {output_dir.absolute()}
train: images/train
val: images/val

nc: {nc}

names:
"""
    for cid in sorted_cids:
        yaml_content += f"  {cid}: '{cid}'\n"

    yaml_content += "\nweights:\n"
    for cid in sorted_cids:
        yaml_content += f"  {cid}: {weights_dict[cid]:.4f}\n"

    new_yaml_path = output_dir / "dataset.yaml"
    with open(new_yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

def split_dataset_pipeline(input_dir: str, output_dir: str, val_ratio: float = 0.2, num_workers: int = 8):
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.exists():
        console.print(f"[bold red]错误：[/bold red]找不到输入目录 {in_path}")
        return

    if out_path.exists():
        console.print(f"[yellow]检测到输出目录已存在，正在清理旧数据: {out_path}[/yellow]")
        shutil.rmtree(out_path)

    for split_type in ["train", "val"]:
        (out_path / "images" / split_type).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split_type).mkdir(parents=True, exist_ok=True)

    all_tasks = []
    table = Table(title="数据集拆分统计", header_style="bold blue")
    table.add_column("类别 ID", justify="center")
    table.add_column("总数量", justify="right")
    table.add_column("Train", justify="right", style="green")
    table.add_column("Val", justify="right", style="magenta")

    with console.status("[bold green]正在扫描数据并分配划分队列..."):
        for class_dir in in_path.iterdir():
            if not class_dir.is_dir(): continue
            
            class_id = class_dir.name
            labels_dir = class_dir / "labels"
            photos_dir = class_dir / "photos"

            if not labels_dir.exists(): continue

            class_pairs = []
            for label_file in labels_dir.glob("*.txt"):
                photo_file = None
                for ext in ['.jpg', '.png', '.jpeg']:
                    temp_photo = photos_dir / (label_file.stem + ext)
                    if temp_photo.exists():
                        photo_file = temp_photo
                        break
                if photo_file:
                    class_pairs.append((label_file, photo_file))

            if not class_pairs: continue

            random.shuffle(class_pairs)
            val_count = int(len(class_pairs) * val_ratio)
            train_count = len(class_pairs) - val_count
            
            table.add_row(class_id, str(len(class_pairs)), str(train_count), str(val_count))

            for i, (label_file, photo_file) in enumerate(class_pairs):
                split_type = "val" if i < val_count else "train"
                
                safe_stem = f"{class_id}_{label_file.stem}"
                dst_label = out_path / "labels" / split_type / f"{safe_stem}.txt"
                dst_photo = out_path / "images" / split_type / f"{safe_stem}{photo_file.suffix}"
                
                all_tasks.append((photo_file, dst_photo, label_file, dst_label))

    if not all_tasks:
        console.print("[yellow]未扫描到有效数据文件。[/yellow]")
        return

    console.print(table)
    console.print(f"\n[bold]总拆分文件数:[/bold] [yellow]{len(all_tasks)}[/yellow] (拆分比例 Train:{1-val_ratio:.2f} / Val:{val_ratio:.2f})\n")

    generate_yaml(in_path, out_path)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console
    )

    with progress:
        main_task = progress.add_task("[cyan]构建并格式化最终数据集...", total=len(all_tasks))
        io_queue = Queue(maxsize=2000)
        
        workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=io_worker, args=(io_queue, progress, main_task), daemon=True)
            t.start()
            workers.append(t)

        for task in all_tasks:
            io_queue.put(task)

        for _ in range(num_workers):
            io_queue.put(None)
            
        for t in workers:
            t.join()

    console.print(Panel(
        f"✨ [bold green]数据集分离准备完毕！[/bold green]\n\n"
        f"输出路径: [underline]{out_path.absolute()}[/underline]\n"
        f"配置清单: [cyan]dataset.yaml[/cyan]\n"
        f"现在请运行 augment.py 针对训练集进行增强。", 
        border_style="green"
    ))

if __name__ == "__main__":
    config_path = "config.yaml"
    val_ratio = 0.2
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            split_cfg = config_data.get('kielas_rm_train', {}).get('dataset', {}).get('split', {})
            if 'val' in split_cfg:
                val_ratio = float(split_cfg['val'])
                console.print(f"[green]已从 {config_path} 加载拆分配置，验证集比例: {val_ratio}[/green]")
    except Exception as e:
        console.print(f"[yellow]读取或解析 {config_path} 失败或未找到配置，将使用默认比例 {val_ratio}[/yellow]")

    split_dataset_pipeline(
        input_dir="./data/balance", 
        output_dir="./data/datasets", 
        val_ratio=val_ratio,
        num_workers=8
    )