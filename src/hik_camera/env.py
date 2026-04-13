import ctypes
import os
from pathlib import Path

# 1. 获取当前脚本所在目录
# .resolve() 获取绝对路径，.parent 获取父目录
current_path = Path(__file__).resolve().parent

# 2. 定义 lib 文件夹路径和 DLL 的完整路径
# 使用 / 运算符进行路径拼接
lib_path = current_path / "hik_lib"
dll_path = lib_path / "MvCameraControl.dll"

# 3. 将 lib 目录添加到 DLL 搜索路径（针对 Python 3.8+）
if hasattr(os, 'add_dll_directory'):
    # os.add_dll_directory 需要传入字符串格式的路径
    os.add_dll_directory(str(lib_path))
else:
    # 兼容旧版本 Python，修改环境变量
    os.environ['PATH'] = f"{lib_path}{os.pathsep}{os.environ['PATH']}"

try:
    # 4. 尝试加载核心库
    # WinDLL 同样需要字符串格式路径
    lib = ctypes.WinDLL(str(dll_path))
    print(f"✅ 核心库加载成功！\n路径: {dll_path}")
except Exception as e:
    print(f"❌ 加载失败。具体路径: {dll_path}")
    print(f"错误信息: {e}")
    print("提示：请检查 hik_lib 目录下是否包含 MvCameraControl.dll 及其所有依赖项。")