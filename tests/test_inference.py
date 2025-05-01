import sys
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
from experiments.inferencer import DTInferencer


from utils.args import parse_args
import argparse
from experiments.trainer import DtTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kitchen.yaml")
    parser.add_argument("--task", type=str)
    parser.add_argument("--model_name", type=str,default="BasicDT")
    args = parser.parse_args()
    args = parse_args(args)
    inferencer = DTInferencer(args)
    inferencer.inference()