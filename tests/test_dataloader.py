import os
import sys
import torch

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

from data import process_dataloader
from utils.args import parse_args
args = parse_args("configs/kitchen.yaml")
dataloader = process_dataloader("kitchen_mixed-v2",args = args)
for batch in dataloader:
    print("")