import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

console = Console()

class WorkflowTerminal:
    def __init__(self):
        # 获取当前根目录路径
        self.root_path = Path(__file__).parent.absolute()
        
        self.menu_options = {
            "1": {"desc": "数据预处理 (Purify/Balance/Split/Augment)", "module": "src.data_process.process"},
            "2": {"desc": "开启模型训练 (Train & Evaluate)", "module": "src.training.train"},
            "3": {"desc": "模型格式导出 (Export to ONNX/TorchScript)", "module": "src.training.export"},
            "4": {"desc": "实时推理演示 (Camera/Video Demo)", "module": "src.demo.demo"},
            "0": {"desc": "退出系统", "module": None}
        }

    def display_menu(self):
        table = Table(show_header=False, border_style="cyan", box=None)
        table.add_column("Index", style="bold green", justify="right")
        table.add_column("Description", style="white")

        for key, value in self.menu_options.items():
            table.add_row(key, value["desc"])

        menu_panel = Panel(
            table,
            title="[bold magenta]RoboMaster 视觉神经网络菜单[/bold magenta]",
            subtitle="[gray]Subprocess Execution Mode[/gray]",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(menu_panel)

    def run_script(self, module_name):
        """使用 subprocess -m 模式调用模块"""
        try:
            # 这里的 module_name 应该是 "src.demo.demo" 这种格式
            # sys.executable 指向当前的 Python 解释器
            subprocess.run([sys.executable, "-m", module_name], check=True)
        except subprocess.CalledProcessError as e:
            console.print(f"[red]模块执行失败，退出码: {e.returncode}[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]操作已被用户中断。[/yellow]")

    def run(self):
        while True:
            console.clear()
            self.display_menu()
            
            choice = Prompt.ask(
                "\n[bold cyan]请选择操作序号[/bold cyan]", 
                choices=list(self.menu_options.keys()), 
                default="1"
            )

            if choice == "0":
                console.print("[yellow]正在退出系统... 祝调试顺利！[/yellow]")
                break

            # 执行对应脚本
            selected_module = self.menu_options[choice]["module"]
            console.print(f"\n[bold reverse] 正在启动独立进程: {self.menu_options[choice]['desc']} [/bold reverse]\n")
            
            self.run_script(selected_module)
            
            Prompt.ask("\n[dim]按回车键返回主菜单...[/dim]")

if __name__ == "__main__":
    terminal = WorkflowTerminal()
    terminal.run()