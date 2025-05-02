import sys
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())


from utils.args import parse_args
import argparse
from experiments.trainer import LptTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kitchen.yaml")
    parser.add_argument("--task", type=str)
    parser.add_argument("--model_name", type=str,default="BasicLPT")
    args = parser.parse_args()
    args = parse_args(args)
    trainer = LptTrainer(args)
    trainer.train()