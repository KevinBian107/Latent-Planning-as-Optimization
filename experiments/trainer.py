from abc import ABC, abstractmethod
import os
import torch
from src.models import get_model
from data import process_dataloader
from logger import MultiLogger

class BaseTrainer(ABC):
    def __init__(self, args):
        self.args = args
        self.dataloader = self._init_dataloder()
        self.logger = self._init_logger()


    @abstractmethod
    def train(self, save_pt=True, save_dir="results/weights", save_checkpoints=True):
        pass

    @abstractmethod
    def _init_model(self):
        """
        _init_model requires to initalize model based on the args
        """
        pass

    def _init_dataloder(self):
        return process_dataloader(env_name=self.args.environment["name"], args = self.args)
    
    def _init_logger(self):
        return MultiLogger(args = self.args, logger_list = self.args.environments["logger"])
    

    @abstractmethod
    def _save_model(self, save_dir):
        pass


class DtTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.model = self._init_model()
        self.dataloader = self._init_dataloder()
        self.logger = self._init_logger()

    def _init_model(self):
        return get_model("BasicDT",**self.args["BasicDT"])
        

    def train(self, save_pt=True, save_dir="results/weights", save_checkpoints=True):
        print("Training DT model...")
        for epoch in range(self.args.training["epochs"]):
            for batch in self.dataloader:
                #TODO
                
                self.logger.log_info({"text":
                                            {
                                                "info":f"[DT] Epoch {epoch + 1} done.",
                                                "loss":loss.item()}
                                            })
                if save_checkpoints:
                    self._save_model(os.path.join(save_dir, f"dt_epoch{epoch + 1}"))

        if save_pt:
            self._save_model(save_dir)

    def _save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "dt_model.pt"))
        print(f"[DT] Model saved to {save_dir}")



class LptTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.model = self._init_model()

    def _init_model(self):
        return get_model("BasicLPT",**self.args["BasicLPT"])


    def train(self, save_pt=True, save_dir="results/weights", save_checkpoints=True):
        print("Training LPT model...")
        #TODO
        for epoch in range(self.args["training"]["epochs"]):
            self.logger.log_info({"text":
                                            {
                                                "info":f"[DT] Epoch {epoch + 1} done.",
                                                "loss":loss.item()}
                                            })

            if save_checkpoints:
                self._save_model(os.path.join(save_dir, f"lpt_epoch{epoch + 1}"))

        if save_pt:
            self._save_model(save_dir)

    def _save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "lpt_model.pt"))
        print(f"[LPT] Model saved to {save_dir}")

