import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from rich.console import Console
from rich.prompt import Confirm
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import matplotlib.pyplot as plt
import matplotlib.patches as patches # 新增: 用于绘制多边形
import numpy as np
from pathlib import Path
import shutil

# 导入模块 (新增了 decode_tensor)
from src.datasets import RMArmorDataset
from src.model import RMDetector, decode_tensor
from src.loss import RMDetLoss

console = Console()

def plot_history(history, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train']) + 1)
    plt.plot(epochs, history['train'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, progress):
    model.train()
    total_loss = 0.0
    task_id = progress.add_task(f"[cyan]Train Epoch {epoch}", total=len(dataloader))
    
    for batch_idx, (imgs, targets, class_ids) in enumerate(dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss, loss_dict = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        progress.update(task_id, advance=1, description=f"[cyan]Train Epoch {epoch} | Loss: {loss.item():.4f}")
        
    progress.remove_task(task_id)
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, progress):
    model.eval()
    total_loss = 0.0
    task_id = progress.add_task(f"[magenta]Val Epoch {epoch}", total=len(dataloader))
    
    for imgs, targets, class_ids in dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss, _ = criterion(preds, targets)
        total_loss += loss.item()
        progress.update(task_id, advance=1)
        
    progress.remove_task(task_id)
    return total_loss / len(dataloader)

@torch.no_grad()
def visualize_predictions(model, dataloader, device, save_dir, prefix, progress, input_size, grid_size, num_samples=5):
    """
    抽取部分数据进行推理，并将真实框与预测框可视化保存。
    已完美支持 13 维 Tensor 到 四点多边形的解码与映射！
    """
    model.eval()
    count = 0
    task_id = progress.add_task(f"[yellow]导出 {prefix} 图像...", total=num_samples)
    
    for imgs, targets, class_ids in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        preds = model(imgs) 
        
        # 核心逻辑：分别解码 GT 和 Pred 张量
        # target_conf threshold 设为 0.9 (因为 target 的 conf 原本就是 1.0)
        gt_dets = decode_tensor(targets, is_pred=False, conf_threshold=0.9, grid_size=grid_size, img_size=input_size)
        # 预测阈值设为 0.5
        pred_dets = decode_tensor(preds, is_pred=True, conf_threshold=0.5, grid_size=grid_size, img_size=input_size)
        
        for i in range(imgs.size(0)):
            if count >= num_samples:
                progress.remove_task(task_id)
                return
            
            img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            
            fig, ax = plt.subplots(1, figsize=(8, 6))
            ax.imshow(img_np)
            
            # --- 绘制真实框 (绿色) ---
            for det in gt_dets[i]:
                # det 结构: [score, x1, y1, x2, y2, x3, y3, x4, y4]
                pts = det[1:].reshape(4, 2)
                polygon_gt = patches.Polygon(pts, linewidth=2, edgecolor='lime', facecolor='none', closed=True)
                ax.add_patch(polygon_gt)
                
            # --- 绘制预测框 (红色) ---
            for det in pred_dets[i]:
                pts = det[1:].reshape(4, 2)
                score = det[0]
                polygon_pred = patches.Polygon(pts, linewidth=2, edgecolor='red', facecolor='none', closed=True, linestyle='--')
                ax.add_patch(polygon_pred)
                ax.text(pts[0,0], pts[0,1]-5, f"{score:.2f}", color='red', fontsize=10, weight='bold')
            
            plt.title(f"{prefix} Set - Sample {count+1}\nGreen: GT | Red: Pred")
            plt.axis('off')
            
            save_path = save_dir / f"{prefix}_sample_{count+1}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            count += 1
            progress.update(task_id, advance=1)

def main():
    config_file = Path("./config.yaml")
    if not config_file.exists():
        console.print(f"[bold red]错误：找不到配置文件 {config_file}[/bold red]")
        return

    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['kielas_rm_train']
    
    train_cfg = cfg['train']
    loss_cfg = cfg['train']['loss']
    data_cfg = cfg['train']['data']

    device = torch.device("cuda" if torch.cuda.is_available() and train_cfg['device'] == 'auto' else train_cfg['device'])
    save_dir = Path(train_cfg.get('save_dir', "./model_res"))
    
    if save_dir.exists() and any(save_dir.iterdir()):
        overwrite = Confirm.ask(f"[bold yellow]输出文件夹 '{save_dir}' 已存在且包含文件，是否清空并刷新？[/bold yellow]")
        if overwrite:
            shutil.rmtree(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            console.print("[green]已清空历史文件夹。[/green]")
        else:
            console.print("[bold red]已取消训练任务以保护现有文件。[/bold red]")
            return
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = train_cfg['epochs']
    history = {'train': [], 'val': []}

    input_size = tuple(train_cfg.get('input_size', [416, 416]))
    grid_size = tuple(train_cfg.get('grid_size', [13, 13]))

    train_loader = DataLoader(
        RMArmorDataset(
            data_cfg['train_img_dir'], 
            data_cfg['train_label_dir'],
            input_size=input_size, 
            grid_size=grid_size
        ),
        batch_size=train_cfg['batch_size'], shuffle=True, num_workers=data_cfg['num_workers']
    )
    val_loader = DataLoader(
        RMArmorDataset(
            data_cfg['val_img_dir'], 
            data_cfg['val_label_dir'],
            input_size=input_size, 
            grid_size=grid_size
        ),
        batch_size=train_cfg['batch_size'], shuffle=False, num_workers=data_cfg['num_workers']
    )

    model = RMDetector().to(device)
    
    criterion = RMDetLoss(
        loss_cfg['lambda_conf'], 
        loss_cfg['lambda_box'], 
        loss_cfg['lambda_pose'],
        grid_size=grid_size
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=float(train_cfg['learning_rate']), weight_decay=float(train_cfg['weight_decay']))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    console.print("[bold green]开始训练...[/bold green]")
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        epoch_task = progress.add_task("[bold green]总体训练进度", total=epochs)
        
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, progress)
            val_loss = validate(model, val_loader, criterion, device, epoch, progress)
            scheduler.step()

            history['train'].append(train_loss)
            history['val'].append(val_loss)
            
            console.print(f"[bold cyan]Epoch {epoch}/{epochs}[/bold cyan] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_dir / "best_model.pth")
            
            progress.update(epoch_task, advance=1)
                
        torch.save(model.state_dict(), save_dir / "last_model.pth")
        plot_history(history, save_dir / "loss_curve.png")
        
        log_file = save_dir / "train_log.txt"
        with log_file.open("w", encoding="utf-8") as f:
            f.write("Epoch\tTrain_Loss\tVal_Loss\n")
            for i in range(len(history['train'])):
                f.write(f"{i+1}\t{history['train'][i]:.6f}\t{history['val'][i]:.6f}\n")

        console.print("\n[bold cyan]正在生成识别效果可视化图片...[/bold cyan]")
        model.load_state_dict(torch.load(save_dir / "best_model.pth"))
        
        # 【修改点】在这里传入了 input_size 和 grid_size
        visualize_predictions(model, train_loader, device, save_dir, prefix="train", progress=progress, input_size=input_size, grid_size=grid_size, num_samples=5)
        visualize_predictions(model, val_loader, device, save_dir, prefix="val", progress=progress, input_size=input_size, grid_size=grid_size, num_samples=5)

    console.print(f"\n[bold green]训练与评估完成！所有结果已保存至: {save_dir.absolute()}[/bold green]")

if __name__ == "__main__":
    main()