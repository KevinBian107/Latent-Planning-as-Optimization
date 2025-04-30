from utils.args import parse_args
from experiments.trainer import LptTrainer,DtTrainer
from experiments.inferencer import LptInferencer, DtInferencer

if __name__ == "__main__":
    args = parse_args()
    if args.task == "train":
        if args.model_name == "BasicDT":
            trainer = DtTrainer(args)
            trainer.train()
            pass
        elif args.model_name == "BasicLPT":
            trainer = LptTrainer(args)
            trainer.train()
        else:
            raise Exception()
    elif args.task == "inference":
        if args.model_name == "BasicDT":
            inferencer = DtInferencer(args)
            inferencer.inference()
        elif args.model_name == "BasicLPT":
            inferencer = LptInferencer(args)
            inferencer.inference()
        else:
            raise Exception()
    else:
        raise Exception()

    

    
    