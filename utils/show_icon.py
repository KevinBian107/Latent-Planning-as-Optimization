"""
没别的用, 写着好玩, 我们可以写个MPI的banner
"""
import time
import subprocess

def launch_tensorboard(logdir="./runs", port=6006, host="localhost"):
    time.sleep(1)  
    subprocess.Popen([
        "tensorboard",
        f"--logdir={logdir}",
        f"--port={port}",
        "--bind_all"
    ])
    print_tensorboard_banner(port=port, host=host)

def print_tensorboard_banner(port=6006, host="localhost"):
    url = f"http://{host}:{port}"
    banner = f"""
\033[1;34m
  ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ ██████╗  ██████╗  █████╗ ██████╗ ██████╗ 
  ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██═██╔══██╗██╔══██║██╔═══██╗██╔══██╗██╔══██╗██╔══██╗
     ██║   █████╗  ██╔██╗ ██║███████╗██║   ██╗██████╔╝██████║ ██║   ██║███████║██████╔╝██║  ██║
     ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔═██╝ ██ ╔═██║██║   ██║██╔══██║██╔═██╝ ██║  ██║
     ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║██████║ ╚██████╔╝██║  ██║██║  ██║██████╔╝
     ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝   ╚═════╝ ╚═╝ ╚═╝╚═╝  ╚═╝  ╚═══╝ 
\033[0m
\033[1;36m╭──────────────────────────────────────────────────────────────────────────────╮
│  \033[1;33m🌐\033[0m  \033[1;37mTensorBoard serving at:\033[0m \033[1;32m{url:<46}\033[1;36m  │
╰──────────────────────────────────────────────────────────────────────────────╯
\033[1;35m
      \033[3;36mVisualize, debug, and understand your ML models\033[0m
      \033[3;90m(Ctrl+C to exit | Open in browser to view dashboards)\033[0m
"""

    print(banner)

