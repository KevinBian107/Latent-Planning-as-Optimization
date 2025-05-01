from abc import ABC, abstractmethod
import os
import torch
from src.models import get_model
from data import process_dataloader
from logger import MultiLogger
import wandb
from tqdm import tqdm

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
        return MultiLogger(args = self.args, logger_list = self.args.training["logger"])
    

    @abstractmethod
    def _save_model(self, save_dir):
        pass


class DtTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model = self._init_model()
        self.device = torch.device(self.args.training["device"])
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                           lr = float(self.args.training["learning_rate"]))

    def _init_model(self):
        return get_model("BasicDT",**self.args.BasicDT)
        

    def train(self, save_pt=True, save_dir="results/weights", save_checkpoints=True):
        self.model.to(self.device)
        total_step = 0
        for epoch in range(self.args.training["epochs"]):
            for i,batch in tqdm(enumerate(self.dataloader),total = len(self.dataloader)):
                state_preds, action_preds, return_preds = self.model(
                timesteps=batch["timesteps"].squeeze(-1).to(self.device),
                states=batch["observations"].to(self.device),
                actions=batch["prev_actions"].to(self.device),
                returns_to_go=batch["return_to_go"].to(self.device)
            )
                self.optimizer.zero_grad()
                loss = torch.nn.MSELoss()(action_preds, batch["actions"].to(self.device))
                
                loss.backward()
                self.optimizer.step()
                total_step += 1
                if i%10 == 0:
                    self.logger.log_info({"step":total_step,
                                        "text":{"training_loss":loss.item()},
                                        "scalars":{"training_loss":loss.cpu()},
                                        })
            if save_checkpoints:
                self._save_model(os.path.join(self.args.path["checkpoint_path"], f"dt_epoch{epoch + 1}"))

        if save_pt:
            self._save_model(self.args.path["weights_path"])

    def _save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "dt_model.pt"))
        print(f"[DT] Model saved to {save_dir}")



class LptTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.model = self._init_model()
        self.device = torch.device(self.args.training["device"])
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                           lr = float(self.args.training["learning_rate"]))

    def _init_model(self):
        return get_model("BasicLPT",**self.args.BasicLPT)


    def train(self, save_pt=True, save_dir="results/weights", save_checkpoints=True):
        self.model.to(self.device)
        total_step = 0
        for epoch in range(self.args.training["epochs"]):
            for i,batch in tqdm(enumerate(self.dataloader),total = len(self.dataloader)):
                batch_inds = torch.arange(batch["observations"].shape[0], device=self.device)
                pred_action, pred_state, pred_reward = self.model(
                    states=batch["observations"].to(self.device),
                    actions=batch["prev_actions"].to(self.device),
                    timesteps=batch["timesteps"].squeeze(-1),
                    rewards=batch["reward"].to(self.device),
                    batch_inds=batch_inds,
                )

                self.optimizer.zero_grad()
                loss_r = torch.nn.MSELoss()(pred_reward, batch["reward"][:, -1, 0].to(self.device))
                loss_a = torch.nn.MSELoss()(pred_action, batch["actions"][:, -1].to(self.device))
                loss = loss_r + loss_a
                loss.backward()
                self.optimizer.step()
                total_step += 1
                if i%10 == 0:
                    self.logger.log_info({"step":total_step,
                                        "text":{"training_loss":loss.item()},
                                        "scalars":{"MSE of Action":loss_a.cpu(),
                                                   "MSE of Rewards":loss_r.cpu(),
                                                   "MSE of Total":loss.cpu()},
                                        })
            if save_checkpoints:
                self._save_model(os.path.join(self.args.path["checkpoint_path"], f"dt_epoch{epoch + 1}"))

        if save_pt:
            self._save_model(self.args.path["weights_path"])

    def _save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "lpt_model.pt"))
        print(f"[LPT] Model saved to {save_dir}")

