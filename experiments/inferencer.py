from abc import ABC, abstractmethod
from src.models import get_model
import torch
import gym
import minari
def make_env(args):
    if args.environment["name"] == "kitchen-mixed-v2":
        dataset = minari.load_dataset('D4RL/kitchen/mixed-v2')
        env = dataset.recover_environment()
    elif args.environment["name"] == "kitchen-complete-v2":
        dataset = minari.load_dataset('D4RL/kitchen/complete-v2')
        env = dataset.recover_environment()
    else:
        raise Exception("Error")
    return env

class BaseInferencer(ABC):
    def __init__(self, args):
        self.args = args
        self.env = make_env(args)
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def inference(self, steps=None):
        pass


class DTInferencer(BaseInferencer):
    def __init__(self,args):
        super().__init__(args)
        self.args = args
        self.model = self._load_model()


    def _load_model(self):
        model = get_model("BasicDT",**self.args.BasicDT)
        return model

    @torch.no_grad()
    def inference(self, steps=None):
        obs = self.env.reset()
        done = False
        t = 0
        while not done and (steps is None or t < steps):
            #
            action = None
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            t += 1



class LPTInferencer(BaseInferencer):
    def _load_model(self):
        model = get_model("BasicLPT",**self.args["BasicLPT"])
        return model

    def inference(self, steps=None):
        #TODO
        obs = self.env.reset()
        t = 0
        done = False

        while not done and (steps is None or t < steps):
            action = ...
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            t += 1




