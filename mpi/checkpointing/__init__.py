from abc import ABC, abstractmethod
import logging
import wandb
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.show_icon import launch_tensorboard
import shutil
import re
from tqdm import tqdm




#Abstract Base Logger
class BaseLogger(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def log_info(self, info: dict):
        pass

    def close(self):
        pass


#Logger to console 
class ConsoleLogger(BaseLogger):
    def __init__(self, args):
        super().__init__(args)
        self.logger = logging.getLogger("console")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_info(self, info: dict):
        if "text" in info:
            for k, v in info["text"].items():
                self.logger.info(f"{k}: {v}")

class FileLogger(BaseLogger):
    def __init__(self, args):
        super().__init__(args)

        log_dir = args.path.get("logs_path", "./results/logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = "experiment.log" #os.path.join(log_dir, args.path.get("experiment_name", "experiment.log"))

        self.logger = logging.getLogger("file_logger")
        self.logger.setLevel(logging.INFO)

        # 避免重复添加 handler（适用于多次初始化）
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_info(self, info: dict):
        if "text" in info:
            for k, v in info["text"].items():
                self.logger.info(f"{k}: {v}")


#log into wandb
class WandbLogger(BaseLogger):
    def __init__(self, args):
        super().__init__(args)
        wandb.init(project=args.path["experiment_name"], 
                   config = eval(f"args.{args.model_name}"))

    def log_info(self, info: dict):
        step = info.get("step", None)

        # scalar
        if "scalars" in info:
            wandb.log(info["scalars"], step=step)

        # image
        if "images" in info:
            image_logs = {k: wandb.Image(v) for k, v in info["images"].items()}
            wandb.log(image_logs, step=step)

        # video
        if "videos" in info:
            for k, v in info["videos"].items():
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                # v.shape: (N, T, C, H, W)
                wandb.log({k: wandb.Video(v, fps=10, format="mp4")}, step=step)

        # weights
        if "weights" in info:
            for name, weight_dict in info["weights"].items():
                for k, v in weight_dict.items():
                    wandb.log({f"{name}/{k}_norm": v.norm().item()}, step=step)


#Log into Tensorboard
# class TensorBoardLogger(BaseLogger):
#     def __init__(self, args):
#         super().__init__(args)
#         log_dir = args.path.get("tensorboard_path", "./runs")
#         if os.path.exists(log_dir):
#             shutil.rmtree(log_dir)
#         self.writer = SummaryWriter(log_dir=log_dir)
#         launch_tensorboard()

#     def log_info(self, info: dict):
#         step = info.get("step", 0)
#         if "scalars" in info:
#             for k, v in info["scalars"].items():
#                 self.writer.add_scalar(k, v, step)

#         # images
#         if "images" in info:
#             for k, v in info["images"].items():
#                 self.writer.add_image(k, v, step)  # expects CHW or NCHW

#         # videos
#         if "videos" in info:
#             for k, v in info["videos"].items():
#                 self.writer.add_video(k, v, step, fps=10)  # expects N,T,C,H,W

#         # weights
#         if "weights" in info:
#             for name, weight_dict in info["weights"].items():
#                 for k, v in weight_dict.items():
#                     self.writer.add_scalar(f"{name}/{k}_norm", v.norm(), step)
class TensorBoardLogger(BaseLogger):
    def __init__(self, args):
        super().__init__(args)
        self.base_log_dir = args.path.get("tensorboard_path", "./runs")
        if os.path.exists(self.base_log_dir):
            shutil.rmtree(self.base_log_dir)
        self.writers = {}  # 存储多个writer的字典
        self.default_writer = SummaryWriter(log_dir=os.path.join(self.base_log_dir, "default"))
        self.writers["default"] = self.default_writer
        launch_tensorboard()

    def _parse_writer_key(self, key):
        """解析key中的(writer_name)部分，返回writer_name和清理后的key"""
        # 使用正则表达式匹配括号及其中内容
        pattern = r"\s*\([^)]*\)\s*"  # 匹配任意空白字符+括号内容+任意空白字符
        writer_name = "default"  # 默认writer
        
        # 查找所有括号内容
        matches = re.findall(r"\(([^)]*)\)", key)
        if matches:
            writer_name = matches[-1]  # 使用最后一个括号内容作为writer名称
        
        # 完全移除所有括号内容及周边空白
        clean_key = re.sub(pattern, "", key).strip()
        return writer_name, clean_key

    def _get_writer(self, writer_name):
        """获取或创建指定的writer"""
        if writer_name not in self.writers:
            log_dir = os.path.join(self.base_log_dir, writer_name)
            os.makedirs(log_dir, exist_ok=True)
            self.writers[writer_name] = SummaryWriter(log_dir=log_dir)
        return self.writers[writer_name]

    def log_info(self, info: dict):
        step = info.get("step", 0)
        
        # 处理scalars
        if "scalars" in info:
            for k, v in info["scalars"].items():
                writer_name, clean_key = self._parse_writer_key(k)
                writer = self._get_writer(writer_name)
                writer.add_scalar(clean_key, v, step)

        # 处理images
        if "images" in info:
            for k, v in info["images"].items():
                writer_name, clean_key = self._parse_writer_key(k)
                writer = self._get_writer(writer_name)
                writer.add_image(clean_key, v, step)  # expects CHW or NCHW

        # 处理videos
        if "videos" in info:
            for k, v in info["videos"].items():
                writer_name, clean_key = self._parse_writer_key(k)
                writer = self._get_writer(writer_name)
                writer.add_video(clean_key, v, step, fps=10)  # expects N,T,C,H,W

        # 处理weights
        if "weights" in info:
            for name, weight_dict in info["weights"].items():
                for k, v in weight_dict.items():
                    writer_name, clean_name = self._parse_writer_key(name)
                    writer = self._get_writer(writer_name)
                    writer.add_scalar(f"{clean_name}/{k}_norm", v.norm(), step)

    def close(self):
        """关闭所有writer"""
        super().close()
        for writer in self.writers.values():
            writer.close()

#A logger provides combine Logger
class MultiLogger(BaseLogger):
    def __init__(self, args, logger_list):
        super().__init__(args)
        LOGGER_MAP = {"file":FileLogger,
                      "wandb":WandbLogger,
                      "tensorboard":TensorBoardLogger,
                      "console":ConsoleLogger}
        
        self.loggers = [LOGGER_MAP[logger_name](args) for logger_name in logger_list]

    def log_info(self, info: dict):
        for logger in self.loggers:
            logger.log_info(info)

    def close(self):
        for logger in self.loggers:
            logger.close()
