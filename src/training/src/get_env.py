import torch

print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"PyTorch 内置 CUDA 版本: {torch.version.cuda}")
print(f"显卡型号: {torch.cuda.get_device_name(0)}")
print(f"算力 (Capability): {torch.cuda.get_device_capability(0)}")