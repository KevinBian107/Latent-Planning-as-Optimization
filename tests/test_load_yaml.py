import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
from utils.args import load_yaml

print(load_yaml("configs/basic.yaml"))