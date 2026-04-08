import os
import cv2
import shutil
import bisect
import subprocess
import rosbag2_py
from concurrent.futures import ProcessPoolExecutor
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import sys
from pathlib import Path
import multiprocessing
import threading

# 引入 rich 相关组件
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn
)

def source_env(script_path):
    """加载 ROS2 接口环境并同步更新 Python 搜索路径"""
    script_path = Path(script_path).expanduser()
    if not script_path.exists():
        print(f"[跳过] 未找到环境脚本: {script_path}")
        return

    try:
        command = f"bash -c 'source {script_path} && env'"
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, text=True)

        for line in proc.stdout:
            if "=" in line:
                key, _, value = line.partition("=")
                value = value.strip()
                os.environ[key] = value

                if key == "PYTHONPATH":
                    for path in value.split(":"):
                        if path and path not in sys.path:
                            sys.path.insert(0, path)

        proc.communicate()
        print(f"[环境] 成功 Source 且已更新 sys.path")
    except Exception as e:
        print(f"[环境] 加载环境失败: {e}")

class RosBagExtractor:
    def __init__(self):
        self.folder_map = {
            "hero_blue_data": 0,      # B1
            "infantry_blue_data": 2,  # B3
            "sentry_blue_data": 5,    # B7
            "hero_red_data": 6,       # R1
            "infantry_red_data": 8,   # R3
            "sentry_red_data": 11     # R7
        }
        self.root_dir = Path(__file__).resolve().parent.parent
        self.original_dir = self.root_dir / "extract_ros2_bag" / "original"
        self.raw_data_dir = self.root_dir / "data" / "raw"

    def prepare_directory(self, target_path: Path):
        """检查并清理目录"""
        try:
            if target_path.exists():
                shutil.rmtree(target_path)
            target_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            pass # 抑制输出以防止破坏 rich 进度条渲染

    def get_reader(self, path: Path):
        storage_options = rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        return reader

    def process_single_bag(self, folder_name, target_id, progress_queue, task_id):
        """处理单个 Bag 的核心逻辑"""
        from rm_interfaces.msg import ArmorsDebugMsg
        
        bag_path = self.original_dir / folder_name
        if not bag_path.exists():
            progress_queue.put({"task_id": task_id, "type": "description", "value": f"[red]未找到目录: {folder_name}"})
            progress_queue.put({"task_id": task_id, "type": "done"})
            return

        base_id_dir = self.raw_data_dir / str(target_id)
        self.prepare_directory(base_id_dir)

        photo_dir = base_id_dir / "photos"
        label_dir = base_id_dir / "labels"
        photo_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        # 更新 UI 状态
        progress_queue.put({"task_id": task_id, "type": "description", "value": f"[cyan]读取标签中: {folder_name}"})
        
        bridge = CvBridge()

        try:
            # ================= 第一步：极速预读取所有标签信息 =================
            label_indices = []
            reader = self.get_reader(bag_path)
            
            storage_filter = rosbag2_py.StorageFilter(topics=['/detector/armors_debug_info'])
            reader.set_filter(storage_filter)
            
            while reader.has_next():
                (topic, data, t) = reader.read_next()
                msg = deserialize_message(data, ArmorsDebugMsg)
                armor_list = []
                for a in msg.armors_debug:
                    armor_list.append({
                        'id': a.armor_id, 'color': a.color,
                        'pts': [a.l_light_up_dx, a.l_light_up_dy, a.l_light_down_dx, a.l_light_down_dy,
                                a.r_light_up_dx, a.r_light_up_dy, a.r_light_down_dx, a.r_light_down_dy]
                    })
                label_indices.append((t, armor_list))

            if not label_indices:
                progress_queue.put({"task_id": task_id, "type": "description", "value": f"[yellow]无标签数据: {folder_name}"})
                progress_queue.put({"task_id": task_id, "type": "done"})
                return

            label_indices.sort(key=lambda x: x[0])
            label_timestamps = [x[0] for x in label_indices]

            # 以提取到的标签总数为进度条总量
            expected_total = len(label_timestamps)
            progress_queue.put({"task_id": task_id, "type": "total", "value": expected_total})
            progress_queue.put({"task_id": task_id, "type": "description", "value": f"[blue]匹配图像中: {folder_name}"})

            # ================= 第二步：流式读取图像并匹配保存 =================
            reader = self.get_reader(bag_path)
            
            image_filter = rosbag2_py.StorageFilter(topics=['/image_raw'])
            reader.set_filter(image_filter)
            
            img_count = 0
            update_batch = 0  # 批量更新进度条以降低通信开销
            
            while reader.has_next():
                (topic, data, t) = reader.read_next()
                
                idx = bisect.bisect_left(label_timestamps, t)
                best_idx = idx
                if idx > 0 and (idx == len(label_timestamps) or abs(t - label_timestamps[idx-1]) < abs(t - label_timestamps[idx])):
                    best_idx = idx - 1

                if abs(t - label_timestamps[best_idx]) < 50_000_000:
                    msg = deserialize_message(data, Image)
                    cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")

                    file_name = f"{img_count:06d}"
                    cv2.imwrite(str(photo_dir / f"{file_name}.jpg"), cv_img)
                    
                    matched_armors = label_indices[best_idx][1]

                    with open(label_dir / f"{file_name}.txt", 'w') as f:
                        for a in matched_armors:
                            pts_str = " ".join(map(str, a['pts']))
                            f.write(f"{a['id']} {a['color']} {pts_str}\n")

                    img_count += 1
                    update_batch += 1
                    
                    # 每 10 帧更新一次 UI，防止进程间通信过载
                    if update_batch >= 10:
                        progress_queue.put({"task_id": task_id, "type": "advance", "value": update_batch})
                        update_batch = 0
                    # 在 while 循环末尾添加：
                    del cv_img
                    del msg
            # 补偿剩余进度
            if update_batch > 0:
                progress_queue.put({"task_id": task_id, "type": "advance", "value": update_batch})

            progress_queue.put({"task_id": task_id, "type": "description", "value": f"[green]已完成 {folder_name} (共{img_count}组)"})
            progress_queue.put({"task_id": task_id, "type": "done"})

        except Exception as e:
            progress_queue.put({"task_id": task_id, "type": "description", "value": f"[red]处理异常: {folder_name}"})
            progress_queue.put({"task_id": task_id, "type": "done"})

    def extract(self):
        print(f"准备并发处理 {len(self.folder_map)} 个数据包 (使用多进程)...")
        
        # 初始化进程安全的队列管理器
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()

        # 配置 rich 进度条的 UI 布局
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold width=30]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )

        def update_progress_thread():
            """在主进程后台持续监听消息并更新 UI"""
            while True:
                try:
                    msg = progress_queue.get()
                    if msg is None:  # 收到退出指令
                        break
                    
                    task_id = msg['task_id']
                    if msg['type'] == 'total':
                        progress.update(task_id, total=msg['value'])
                    elif msg['type'] == 'advance':
                        progress.advance(task_id, advance=msg['value'])
                    elif msg['type'] == 'description':
                        progress.update(task_id, description=msg['value'])
                    elif msg['type'] == 'done':
                        # 若任务完成但进度未满，直接填满
                        current_task = progress.tasks[task_id]
                        if current_task.total is None:
                            progress.update(task_id, total=100, completed=100)
                        else:
                            progress.update(task_id, completed=current_task.total)
                except Exception:
                    pass

        # 启动 rich 进度条上下文
        with progress:
            # 预设任务占位符
            tasks = {}
            for folder_name in self.folder_map.keys():
                # 初始 total=None，呈现不确定状态，直到完成标签读取再设定总量
                task_id = progress.add_task(f"[cyan]等待中: {folder_name}", total=None)
                tasks[folder_name] = task_id

            # 启动监听线程
            listener = threading.Thread(target=update_progress_thread)
            listener.start()

            with ProcessPoolExecutor(max_workers=6) as executor:
                futures = []
                for folder_name, target_id in self.folder_map.items():
                    futures.append(
                        executor.submit(
                            self.process_single_bag, 
                            folder_name, 
                            target_id, 
                            progress_queue, 
                            tasks[folder_name]
                        )
                    )
                # 等待所有进程结束
                for future in futures:
                    future.result()

            # 关闭监听线程
            progress_queue.put(None)
            listener.join()

if __name__ == '__main__':
    source_path = "~/DT46_V/install/setup.bash"

    if "DT46_V" not in os.environ.get("LD_LIBRARY_PATH", ""):
        print("检测到环境未完全加载，正在重新注入环境变量并重启脚本...")
        source_env(source_path)
        os.execv(sys.executable, ['python3'] + sys.argv)

    extractor = RosBagExtractor()
    extractor.extract()