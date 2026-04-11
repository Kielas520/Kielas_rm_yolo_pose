import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from rich.console import Console

# 导入之前写好的模块
from src.datasets import RMArmorDataset
from src.model import RMDetector
from src.loss import RMDetLoss

console = Console()

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (imgs, targets, class_ids) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        # 1. 梯度清零
        optimizer.zero_grad()
        
        # 2. 前向传播
        preds = model(imgs)
        
        # 3. 计算损失
        loss, loss_dict = criterion(preds, targets)
        
        # 4. 反向传播与优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 打印日志 (每 10 个 Batch 打印一次)
        if batch_idx % 10 == 0:
            console.print(f"[Train] Epoch: {epoch} | Batch: {batch_idx}/{len(dataloader)} | "
                          f"Loss: {loss.item():.4f} (Conf: {loss_dict['loss_conf']:.4f}, "
                          f"Box: {loss_dict['loss_box']:.4f}, Pose: {loss_dict['loss_pose']:.4f})")
            
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    for imgs, targets, class_ids in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        preds = model(imgs)
        loss, _ = criterion(preds, targets)
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def main():
    # -----------------------------------------
    # 0. 解析 YAML 配置
    # -----------------------------------------
    config_path = "./config.yaml"
    if not os.path.exists(config_path):
        console.print(f"[bold red]错误：未找到配置文件 {config_path}[/bold red]")
        return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    base_cfg = cfg.get('kielas_rm_train', {})
    train_cfg = base_cfg.get('train', {})
    data_cfg = base_cfg.get('train', {}).get('data', {})
    loss_cfg = base_cfg.get('train', {}).get('loss', {})

    # -----------------------------------------
    # 1. 超参数设置
    # -----------------------------------------
    device_cfg = train_cfg.get('device', 'auto')
    if device_cfg == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    console.print(f"[bold green]Using device: {device}[/bold green]")
    
    batch_size = train_cfg.get('batch_size', 4)
    epochs = train_cfg.get('epochs', 1)
    learning_rate = float(train_cfg.get('learning_rate', 1e-3))
    weight_decay = float(train_cfg.get('weight_decay', 1e-4))
    
    save_dir = train_cfg.get('save_dir', "./weights")
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------------------
    # 2. 初始化 Dataset 和 DataLoader
    # -----------------------------------------
    train_dataset = RMArmorDataset(
        img_dir=data_cfg.get('train_img_dir', "./data/datasets/images/train"), 
        label_dir=data_cfg.get('train_label_dir', "./data/datasets/labels/train")
    )
    val_dataset = RMArmorDataset(
        img_dir=data_cfg.get('val_img_dir', "./data/datasets/images/val"), 
        label_dir=data_cfg.get('val_label_dir', "./data/datasets/labels/val")
    )
    
    num_workers = data_cfg.get('num_workers', 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # -----------------------------------------
    # 3. 初始化 模型、损失函数和优化器
    # -----------------------------------------
    model = RMDetector().to(device)
    
    lambda_conf = float(loss_cfg.get('lambda_conf', 1.0))
    lambda_box = float(loss_cfg.get('lambda_box', 2.0))
    lambda_pose = float(loss_cfg.get('lambda_pose', 1.0))
    criterion = RMDetLoss(lambda_conf=lambda_conf, lambda_box=lambda_box, lambda_pose=lambda_pose).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # -----------------------------------------
    # 4. 主训练循环
    # -----------------------------------------
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        console.rule(f"Epoch {epoch}/{epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        console.print(f"[bold cyan]Epoch {epoch} Summary:[/bold cyan] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            console.print(f"[bold red]Best model saved to {save_path}[/bold red]")
            
    # 保存最后一个 epoch 的模型
    torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))
    console.print("[bold green]Training Completed![/bold green]")

if __name__ == "__main__":
    main()