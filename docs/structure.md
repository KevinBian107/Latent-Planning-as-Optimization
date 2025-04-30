# Repo Structure
```bash
├── __init__.py  
├── configs  
│   └── kitchen.yaml  
│   └── constant.py
├── docs  
├── experiment  
│   ├── trainer.py
│   └── inferencer.py
│      
├── data  
│   ├── dataloader  
│   └── dataset  
├── src  
│   ├── models  
│   │   ├── unet1d.py  
│   │   ├── decision_transformer.py  
│   │   ├── conditional_decision_transformer.py  
│   │   └── LPT.py      
│   └── layers  
│       ├── block.py  
│       ├── attention.py  
│       └── ...  
├── utils  
│   ├── args.py  
│   └── how_far_we_go.py  
├── logs  
│   ├── tensorboard  
│   └── wandb  
├── results
│   ├── videos 
│   ├── logs
│   ├── checkpoints
│   └── weights
```
## 创建/扩展新的辅助类
### 1. 新建的子类需要继承原有的抽象类
### 2. 当创造子类父类继承关系时， 如果代码量不大的情况下， 可以考虑放在同一个文件。 但是如果存在的子类超过八个或者代码过长，考虑将其原本的文件变为folder，将各个类放在目录下不同的python文件里，并在目录的__init__.py里面import他们.
### eg: 目前结构为：experiments
###                  |- trainer.py
### 使用 from experiments.trainer import LptTrainer 调用trainer
###
### 当trainer多于八种，代码量超过300行时：
### 项目结构改为 experiments
###                  |- trainer 
###                        |- __init__.py
###                        |- LptTrainer.py
###                        |- PtTrainer.py
###                              ...
### 这样其他文件中的from experiments.trainer import LptTrainer 依然有效
### 3. 对于新补充的代码，在/test 加上它的测试，包括它的定义，调用

## 变量定义
### 如果需要在后续改变的变量：写入yaml的字段中
### 如果是类似普朗克常量或者Pi之类的不变量， 考虑以常量形式定义在configs/constant.py中
### 为了测试方便， 仅在tests 文件夹中的测试文件允许magic number



## 新建模型类
### 当新建模型类时，如果它是一个直接引用和训练的模型（basicPT和LPT），而非模型结构的一部分（例如unet）。 你需要
### 1. 用@register_model 来为它注册一个名字 假设为name_A
### 2. 在config文件中添加name_A 所使用的各种参数
### 3. 为name_A 添加对应的data_loader， 考虑到不同模型也许需要不同的损失函数，optimizer等达到最优
### 对于新补充的模型，在/test 中加上它的测试，包括它的定义，调用和在toy_dataset上的收敛


## 新建环境
### 新建环境需要在data/dataset.py 里面创建新的datasets，并在inferencer.py里面补充make_env的方法


## 新建helper_function
### 如果helper_function 和 helper_class 数量较小，并且只被一个py文件调用，可以放在该文件的开头, 否则放在util里面使用        

## 命名风格
### py文件，function, 和变量采用下划线命名风格， 类名采用驼峰命名法（例如：EnvMaker或者 LptTrainer）
