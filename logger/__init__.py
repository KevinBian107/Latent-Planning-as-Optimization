from abc import ABC, abstractmethod
import logging
import wandb
import os
import torch
from torch.utils.tensorboard import SummaryWriter


#Abstract Base Logger
class BaseLogger(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def log_info(self, info: dict):
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

        log_dir = args.get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, args.get("log_filename", "experiment.log"))

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
        wandb.init(project=args.experiment_name, args=args)

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
class TensorBoardLogger(BaseLogger):
    def __init__(self, args):
        super().__init__(args)
        self.writer = SummaryWriter(log_dir=args.get("tb_log_dir", "./runs"))

    def log_info(self, info: dict):
        step = info.get("step", 0)
        if "scalars" in info:
            for k, v in info["scalars"].items():
                self.writer.add_scalar(k, v, step)

        # images
        if "images" in info:
            for k, v in info["images"].items():
                self.writer.add_image(k, v, step)  # expects CHW or NCHW

        # videos
        if "videos" in info:
            for k, v in info["videos"].items():
                self.writer.add_video(k, v, step, fps=10)  # expects N,T,C,H,W

        # weights
        if "weights" in info:
            for name, weight_dict in info["weights"].items():
                for k, v in weight_dict.items():
                    self.writer.add_scalar(f"{name}/{k}_norm", v.norm(), step)


#A logger provides combine Logger
class MultiLogger(BaseLogger):
    def __init__(self, args, logger_list):
        super().__init__(args)
        self.loggers = logger_list

    def log_info(self, info: dict):
        for logger in self.loggers:
            logger.log_info(info)
