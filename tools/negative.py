import yaml
import shutil
from pathlib import Path
from rich.console import Console
from rich.progress import track

console = Console()

def main():
    config_file = Path("config.yaml")
    if not config_file.exists():
        console.print("[bold red]错误：找不到 config.yaml 文件[/bold red]")
        return

    # 1. 解析配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    train_cfg = cfg.get('kielas_rm_train', {})
    dataset_cfg = train_cfg.get('dataset', {})
    
    # 获取负样本类别 ID
    neg_class_id = train_cfg.get('train', {}).get('negative_class_id', 12)
    
    # 获取输入路径：这里默认读取 yaml 中的背景图目录，你也可以根据实际情况修改这个键值
    input_dir_str = dataset_cfg.get('augment', {}).get('bg_dir', './background')
    input_dir = Path(input_dir_str)
    
    # 获取输出根路径：从 dataset 的配置中提取，例如从 "./data/balance" 中提取出 "./data"
    # 如果没找到，默认使用 "./data"
    balance_dir_str = dataset_cfg.get('balance_dir', './data/balance')
    output_root = Path(balance_dir_str).parent 
    
    if not input_dir.exists():
        console.print(f"[bold red]错误：输入图库目录 {input_dir} 不存在，请检查！[/bold red]")
        return

    # 2. 构建标准的输出目录结构: output_root/id/photos 和 output_root/id/labels
    photos_dir = output_root / str(neg_class_id) / "photos"
    labels_dir = output_root / str(neg_class_id) / "labels"
    
    # 清空可能存在的旧数据，或者直接追加（这里选择如果存在则不报错，直接覆盖/追加）
    photos_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 3. 收集所有支持的图片文件
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_paths = [p for p in input_dir.rglob('*') if p.suffix.lower() in valid_exts]
    
    if not img_paths:
        console.print(f"[bold yellow]警告：在 {input_dir} 中没有找到任何图片文件[/bold yellow]")
        return

    console.print(f"[cyan]找到 {len(img_paths)} 张背景图，准备生成纯负样本数据集 (ID: {neg_class_id})...[/cyan]")

    # 4. 执行文件拷贝与标签生成
    for i, img_path in enumerate(track(img_paths, description="[green]生成负样本中...")):
        # 统一格式化文件名为 5 位序号：00000, 00001 ...
        file_name = f"{i:05d}"
        
        dst_img_path = photos_dir / f"{file_name}{img_path.suffix.lower()}"
        dst_label_path = labels_dir / f"{file_name}.txt"
        
        # 拷贝图像文件
        shutil.copy(img_path, dst_img_path)
        
        # 写入纯 ID 标签文件 (无 bbox/pose 坐标)
        with open(dst_label_path, 'w', encoding='utf-8') as f:
            f.write(f"{neg_class_id}\n")

    console.print("[bold green]✅ 负样本数据集构建完成！[/bold green]")
    console.print(f" -> 图片已保存至: {photos_dir.absolute()}")
    console.print(f" -> 标签已保存至: {labels_dir.absolute()}")

if __name__ == "__main__":
    main()