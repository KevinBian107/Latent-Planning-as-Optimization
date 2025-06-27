from utils.args import parse_args
from agent.trainer import LptTrainer, DtTrainer
from agent.inferencer import LPTInferencer, DTInferencer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,default="mpi/configs/maze2d.yaml")
    parser.add_argument("--task", default="training")#choices=["inference","training"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type = str, default="BasicLPT")
    parser.add_argument("--train_type", type = str, default="mixed")
    args = parser.parse_args()
    config = parse_args(args)

    
    if config.model_name == "BasicLPT":
        if config.task == "training":
            trainer = LptTrainer(config)
            
            if config.train_type == "mixed":
                trainer.mixed_train()
                
        elif config.task == "inference":
            inferencer = LPTInferencer(config)
            inferencer.inference(100)
    
    elif config.model_name == "BasicDT":
        if config.task == "training":
            trainer = DtTrainer(config)
            
            if config.train_type == "mixed":
                trainer.mixed_train()
            
        elif config.task == "inference":
            inferencer = DTInferencer(config)
            inferencer.inference(1000)
    else:
        raise Exception("no model found")


    


    

    
    