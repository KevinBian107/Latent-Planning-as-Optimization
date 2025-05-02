import sys
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
from logger import TensorBoardLogger,WandbLogger,FileLogger,ConsoleLogger,MultiLogger
from utils.args import parse_args
import argparse
import torch
import torch.nn as nn
import wandb
def test_file_logger():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kitchen.yaml")
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    args = parse_args(args)
    logger_file = FileLogger(args)
    logger_file.log_info({"text":{"loss":"0.25"}})

def test_console_logger():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kitchen.yaml")
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    args = parse_args(args)
    logger_file = ConsoleLogger(args)
    logger_file.log_info({"text":{"loss":"0.25"}})

def test_wandb_logger():
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = SimpleMLP()
    state_dict = model.state_dict()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kitchen.yaml")
    parser.add_argument("--task", type=str)
    parser.add_argument("--model_name",type=str, default="BasicLPT")
    args = parser.parse_args()
    args = parse_args(args)
    logger_file = WandbLogger(args)
    for i in range(1,11,1):
        if i%10 == 0:
            logger_file.log_info({"step":i,
                                "text":{"training_loss":0.25},
                                "weights":{"layer_norms":state_dict},
                                "scalars":{"training_loss":0.25*(5/i),
                                        "validation_loss":0.35*(5/i),
                                        "test_loss":0.45*(5/i)},
                                "images":{"random_noise":torch.randn((3,255,255))},
                                "videos":{"random_noise_animation":torch.rand(1, 60, 3, 64, 64) }
                                })
            
        else:
            logger_file.log_info({"step":i,
                                "text":{"training_loss":0.25},
                                "weights":{"layer_norms":state_dict},
                                "scalars":{"training_loss":0.25*(5/i),
                                        "validation_loss":0.35*(5/i),
                                        "test_loss":0.45*(5/i)},
                                })
    wandb.finish()

def test_tensorboard_logger():
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = SimpleMLP()
    state_dict = model.state_dict()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kitchen.yaml")
    parser.add_argument("--task", type=str)
    parser.add_argument("--model_name",type=str, default="BasicLPT")
    args = parser.parse_args()
    args = parse_args(args)
    logger_file = TensorBoardLogger(args)
    for i in range(1,11,1):
        if i%10 == 0:
            logger_file.log_info({"step":i,
                                "text":{"training_loss":0.25},
                                "weights":{"layer_norms":state_dict},
                                "scalars":{"training_loss":torch.tensor(0.25*(5/i)),
                                        "validation_loss":torch.tensor(0.35*(5/i)),
                                        "test_loss":torch.tensor(0.45*(5/i))},
                                "images":{"random_noise":torch.randn((3,255,255))},
                                "videos":{"random_noise_animation":torch.rand(1, 60, 3, 64, 64) }
                                })
            
        else:
            logger_file.log_info({"step":i,
                                "text":{"training_loss":0.25},
                                "weights":{"layer_norms":state_dict},
                                "scalars":{"training_loss":torch.tensor(0.25*(5/i)),
                                        "validation_loss":torch.tensor(0.35*(5/i)),
                                        "test_loss":torch.tensor(0.45*(5/i))},
                                })
    
def test_multi_logger():
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = SimpleMLP()
    state_dict = model.state_dict()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kitchen.yaml")
    parser.add_argument("--task", type=str)
    parser.add_argument("--model_name",type=str, default="BasicLPT")
    args = parser.parse_args()
    args = parse_args(args)
    logger_file = MultiLogger(args = args,logger_list=["file","tensorboard","wandb","console"])
    for i in range(1,11,1):
        if i%10 == 0:
            logger_file.log_info({"step":i,
                                "text":{"training_loss":0.25*(5/i)},
                                "weights":{"layer_norms":state_dict},
                                "scalars":{"training_loss":0.25*(5/i),
                                        "validation_loss":0.35*(5/i),
                                        "test_loss":0.45*(5/i)},
                                "images":{"random_noise":torch.randn((3,255,255))},
                                })
            
        else:
            logger_file.log_info({"step":i,
                                "text":{"training_loss":0.25*(5/i)},
                                "weights":{"layer_norms":state_dict},
                                "scalars":{"training_loss":0.25*(5/i),
                                        "validation_loss":0.35*(5/i),
                                        "test_loss":0.45*(5/i)},
                                })


if __name__ == "__main__":
    test_multi_logger()

