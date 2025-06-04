from abc import ABC, abstractmethod
import os
import torch
import minari

from tqdm import tqdm
import wandb
import pdb

from agent.src.models import get_model
from checkpointing import MultiLogger

from data.processors import SequenceProcessor, KitchenSegmenter
from data.data_processor import DataProcessor
from data.dataset import MinariTrajectoryDataset
from data.batch_generator import (TaskBatchGenerator, SingleTaskBatchGenerator)


def process_dataloader(env_name: str, env_key: str, context_len, args):

    downloaded_data = minari.load_dataset(env_name, download=True)
    dataset = MinariTrajectoryDataset(dataset=downloaded_data)

    sequence_processor = SequenceProcessor(
        context_len = context_len,
        device = args.training["device"]
    )

    sequence_processor.fit(dataset)
    
    # a better way to hand env and data processor mapping?
    env_processors = {
        "kitchen-complete-v2": lambda: {
            'pipeline_name': 'multi_task_segment',
            'processors': {
                'sequence_processor': sequence_processor,
                'segmenter_processor': KitchenSegmenter(
                    task_goal_keys=['microwave', 'kettle', 'light switch', 'slide cabinet'],
                    proximity_thresholds={
                        'microwave': 0.2,
                        'kettle': 0.3,
                        'light switch': 0.2,
                        'slide cabinet': 0.2
                    },
                    stability_duration=20
                )
            },
            'batch_style': 'task_batch_generator'
        },

        "kitchen-mixed-v2": lambda: {
            'pipeline_name': 'single_task',
            'processors': {'sequence_processor': sequence_processor}, 
            'batch_style': 'single_task_batch_generator'
        },

        "halfcheetah-expert-v0": lambda: {
            'pipeline_name': 'single_task',
            'processors': {'sequence_processor': sequence_processor}, 
            'batch_style': 'single_task_batch_generator'
        }, 

    }
    
    # Determine environment type
    assert env_key in env_processors, f"Environment key '{env_key}' not found in processors mapping."
        
    # Process the dataset using the appropriate configuration
    data_processor = DataProcessor()
    processor_config = env_processors[env_key]()
    processed_data = data_processor.process_dataset(
        dataset=dataset,
        pipeline_name=processor_config['pipeline_name'],
        processors=processor_config['processors']
    )
    batch_style = processor_config['batch_style']

    if batch_style == 'task_batch_generator':
        batch_generator = TaskBatchGenerator(
            processed_data=processed_data,
            device=args.training["device"],
            batch_size=args.training["batch_size"]
        )

        task_name = args.training.get("task_name", "microwave")
        task_name = args.training.get("task_name", env_name)
        task_name = 'microwave'
        # FIXME: 
        # idealy, we will have a list of task name passed in to the train function
        # the list of task name will be iterated during training loop and get_batch() will be called at the training time
        # process_dataloader() function should return the batch generator object itself, not the get_batch function
        return batch_generator.get_batch(task_name)
    
    if batch_style == 'single_task_batch_generator':
        batch_generator = SingleTaskBatchGenerator(
            processed_data=processed_data,
            device=args.training["device"],
            batch_size=args.training["batch_size"]
        )

        return batch_generator.get_batch()

    # if "kitchen" in env_name:
    #     kitchen_segmenter = KitchenSegmenter(
    #         task_goal_keys=['microwave', 'kettle', 'light switch', 'slide cabinet'],
    #         proximity_thresholds={
    #             'microwave': 0.2,
    #             'kettle': 0.3,
    #             'light switch': 0.2,
    #             'slide cabinet': 0.2
    #         },
    #         stability_duration=20
    #     )

    #     data_processor = DataProcessor()
    #     processed_data = data_processor.process_dataset(
    #         dataset=dataset,
    #         pipeline_name='multi_task_segment',
    #         processors={
    #             'sequence_processor': sequence_processor,
    #             'segmenter_processor': kitchen_segmenter
    #         }
    #     )
        
    #     batch_generator = TaskBatchGenerator(
    #     processed_data=processed_data,
    #     device=args.training["device"],
    #     batch_size=args.training["batch_size"]
    #     )

    #     # You can dynamically set the task here if needed
    #     task_name = args.training.get("task_name", "microwave")
    #     task_name = args.training.get("task_name", env_name)
    
    # if "halfcheetah" in env_name:
    #     data_processor = DataProcessor()
    #     processed_data = data_processor.process_dataset(
    #         dataset=dataset,
    #         pipeline_name='single_task',
    #         processors={'sequence_processor': sequence_processor}
    #     )

    
    # return batch_generator


class BaseTrainer(ABC):
    def __init__(self, args):
        self.args = args
        self.dataloader = self._init_dataloder()
        self.logger = self._init_logger()


    @abstractmethod
    def mixed_train(self, save_pt=True, save_dir="results/weights", save_checkpoints=True):
        pass

    @abstractmethod
    def _init_model(self):
        """
        _init_model requires to initalize model based on the args
        """
        pass

    def _init_dataloder(self):
        env_mapping = {
            "kitchen-mixed-v2": "D4RL/kitchen/mixed-v2",
            "kitchen-complete-v2": "D4RL/kitchen/complete-v2",
            "halfcheetah-expert-v0": "mujoco/halfcheetah/expert-v0"
        }

        env_key = self.args.environment["name"]
        if env_key not in env_mapping:
            raise Exception(f"{env_key} not found in environment mapping.")

        env_name = env_mapping[env_key]
        return process_dataloader(
            env_name=env_name,
            env_key=env_key,
            context_len= self.args.environment["context_len"],
            args=self.args
        )
    
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
        

    def mixed_train(self, save_pt=True, save_dir="results/weights", save_checkpoints=True):
        self.model.to(self.device)
        total_step = 0
        for epoch in range(self.args.training["epochs"]):
            # for i, batch in tqdm(enumerate(self.dataloader),total = len(self.dataloader)):
            for i, batch in enumerate(tqdm(self.dataloader)):
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


    def mixed_train(self, save_pt=True, save_dir="results/weights", save_checkpoints=True):
        self.model.to(self.device)
        total_step = 0
        for epoch in range(self.args.training["epochs"]):
            # for i,batch in tqdm(enumerate(self.dataloader),total = len(self.dataloader)):
            for i, batch in enumerate(tqdm(self.dataloader)):
                batch_inds = torch.arange(batch["observations"].shape[0], device=self.device)
                pred_action, pred_state, pred_reward = self.model(
                    states=batch["observations"].to(self.device),
                    actions=batch["prev_actions"].to(self.device),
                    timesteps=batch["timesteps"].squeeze(-1),
                    rewards=torch.sum(batch["reward"],dim = 1).to(self.device),
                    batch_inds=batch_inds,
                )

                self.optimizer.zero_grad()
                loss_r = torch.nn.MSELoss()(pred_reward, torch.sum(batch["reward"], dim = 1).squeeze(1).to(self.device))
                loss_a = torch.nn.MSELoss()(pred_action, batch["actions"][:, -1].to(self.device))
                loss = 0.25 * loss_r + loss_a
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
            self._save_model(self.args.path["weights_path"],weight_only=False)

    def _save_model(self, save_dir, weight_only = True):
        os.makedirs(save_dir, exist_ok=True)
        if weight_only:
            torch.save(self.model.state_dict(), os.path.join(save_dir, "lpt_model.pt"))
        else:
            torch.save(self.model,os.path.join(save_dir, "lpt_model.pt"))
        print(f"[LPT] Model saved to {save_dir}")
