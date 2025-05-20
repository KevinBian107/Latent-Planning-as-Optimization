from utils.args import parse_args
from experiments.trainer import LptTrainer,DtTrainer
from experiments.inferencer import LPTInferencer,DTInferencer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,default="configs/kitchen.yaml")
    parser.add_argument("--device", type=str,default="mps")
    parser.add_argument("--task", choices=["inference","training"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type = str, default="BasicDT")
    args = parser.parse_args()
    config = parse_args(args)
    config
    if config.model_name == "BasicLPT":
        if config.task == "train":
            trainer = LptTrainer(config)
            trainer.train()
        elif config.task == "inference":
            inferencer = LPTInferencer(config)
            inferencer.inference(100)
    elif config.model_name == "BasicDT":
        if config.task == "train":
            pass
        elif config.task == "inference":
            inferencer = DTInferencer(config)
            inferencer.inference(1000)
    else:
        raise Exception("no model found")


    


    

    
    