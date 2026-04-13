import yaml
import cv2
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.status import Status
from rich.panel import Panel

# 1. 导入模型组件
# 路径：src -> training -> src -> model.py
from src.training.src.model import RMDetector, decode_tensor

# 2. 导入海康相机
# 路径：src -> hik_camera -> hik_camera.py
from src.hik_camera.hik_camera import HikCamera

console = Console()

class InferenceEngine:
    # ... (保持你原有的 InferenceEngine 类代码不变)
    def __init__(self, cfg):
        self.type = cfg['model_type'].lower()
        self.device = torch.device(cfg.get('device', 'cpu'))
        self.model_path = Path(cfg['model_path'])
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {self.model_path.absolute()}")

        with Status(f"[bold cyan]正在加载 {self.type.upper()} 引擎...", console=console):
            if self.type == "onnx":
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
                self.session = ort.InferenceSession(str(self.model_path), providers=providers)
                self.input_name = self.session.get_inputs()[0].name
            elif self.type == "pytorch":
                self.model = RMDetector().to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
            elif self.type == "torchscript":
                self.model = torch.jit.load(str(self.model_path), map_location=self.device)
                self.model.eval()

    def __call__(self, x_tensor):
        if self.type == "onnx":
            x_np = x_tensor.cpu().numpy()
            out_np = self.session.run(None, {self.input_name: x_np})[0]
            return torch.from_numpy(out_np).to(self.device)
        else:
            with torch.no_grad():
                return self.model(x_tensor)

def draw_and_extract(frame, dets, orig_shape, input_size):
    # ... (保持你原有的绘制函数不变)
    orig_h, orig_w = orig_shape
    scale_x, scale_y = orig_w / input_size[0], orig_h / input_size[1]
    color_red, color_black = (0, 0, 255), (0, 0, 0)
    info_list = []

    for det in dets:
        score, cls_id = det[0], int(det[1])
        pts = det[2:].reshape(4, 2)
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        pts = pts.astype(np.int32)
        
        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        info_list.append(f"{cls_id}, ({cx}, {cy})")
        
        for p in pts: cv2.circle(frame, tuple(p), 4, color_red, -1)
        cv2.line(frame, tuple(pts[0]), tuple(pts[1]), color_red, 2)
        cv2.line(frame, tuple(pts[2]), tuple(pts[3]), color_red, 2)
        
        text = f"ID:{cls_id} | {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = cx + 20, cy - 20
        cv2.rectangle(frame, (tx-3, ty-th-3), (tx+tw+3, ty+bl+3), color_red, 1)
        cv2.rectangle(frame, (tx-2, ty-th-2), (tx+tw+2, ty+bl+2), color_black, -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_red, 1)
        cv2.line(frame, (cx, cy), (tx-3, ty-th//2), color_red, 1)

    return frame, info_list

def main():
    config_file = Path("./config.yaml")
    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['kielas_rm_demo']
    
    input_sz, grid_sz = tuple(cfg['input_size']), tuple(cfg['grid_size'])
    conf_t, nms_t = cfg['conf_threshold'], cfg['nms_iou_threshold']
    device = torch.device(cfg.get('device', 'cpu'))
    
    engine = InferenceEngine(cfg)
    
    # --- 相机工厂逻辑 ---
    cam_type = cfg.get('camera_type', 'usb').lower()
    cam_index = cfg.get('camera_index', 0)
    
    if cam_type == "hik":
        cap = HikCamera(cam_index)
        if not cap.open():
            console.print("[bold red]无法打开海康相机[/bold red]")
            return
        # 初始曝光设置
        cap.set_exposure(cfg.get('hik_exposure', 5000))
        current_exposure = cap.get_exposure()
    else:
        cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 兼容部分 USB 相机手动模式
        current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)

    console.print(Panel(
        f"[bold cyan]相机模式: {cam_type.upper()}[/bold cyan]\n"
        "[bold yellow]W / S[/bold yellow] : 调整曝光\n"
        "[bold yellow]Q[/bold yellow] : 退出程序", title="Camera System"
    ))

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                continue

            orig_shape = raw_frame.shape[:2]
            st = cv2.getTickCount()
            
            # 预处理并推理
            img = cv2.resize(raw_frame, input_sz)
            img_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            preds = engine(img_tensor)
            dets = decode_tensor(
                preds, 
                is_pred=True, 
                conf_threshold=conf_t, 
                nms_iou_threshold=nms_t, 
                grid_size=grid_sz, 
                img_size=input_sz
            )
            
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - st)
            
            # 绘制
            out_frame, info = draw_and_extract(raw_frame, dets[0], orig_shape, input_sz)
            
            # 叠加状态
            status_text = f"FPS: {fps:.1f} | Exposure: {current_exposure:.0f}"
            cv2.putText(out_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Detection", out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                break
            elif key == ord('w'):
                if cam_type == "hik":
                    current_exposure += 500
                    cap.set_exposure(current_exposure)
                else:
                    current_exposure += 1
                    cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
            elif key == ord('s'):
                if cam_type == "hik":
                    current_exposure = max(100, current_exposure - 500)
                    cap.set_exposure(current_exposure)
                else:
                    current_exposure -= 1
                    cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)

    finally:
        # 统一释放逻辑
        if cam_type == "hik":
            cap.close()
        else:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()