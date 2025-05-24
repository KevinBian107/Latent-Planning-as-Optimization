# Repo Structure

We provide the following guide for showing teh structure of our repository:

```bash
mpi/
├── agent/
│   ├── __pycache__/
│   ├── legacy/
│   └── src/
│       ├── layers/
│       │   ├── __pycache__/
│       │   ├── __init__.py
│       │   ├── attention.py
│       │   └── block.py
│       ├── models/
│       │   ├── __pycache__/
│       │   ├── __init__.py
│       │   ├── alpha_networks.py
│       │   ├── anil_generator.py
│       │   ├── anil.py
│       │   ├── conditional_decision_transformer.py
│       │   ├── decision_transformer.py
│       │   ├── lpt_variational.py
│       │   ├── lpt.py
│       │   ├── mpill.py
│       │   ├── unet1d.py
│       │   └── vae1d.py
│       ├── util_function/
│       │   ├── __pycache__/
│       │   ├── __init__.py
│       │   └── model_registry.py
│       ├── inferencer.py
│       └── trainer.py
├── checkpointing/
│   └── __init__.py
├── configs/
├── data/
│   ├── __pycache__/
│   ├── legacy/
│   ├── __init__.py
│   ├── batch_generator.py
│   ├── data_processor.py
│   ├── dataset.py
│   ├── example_use.py
│   └── processors.py
├── utils/
│   ├── __pycache__/
│   ├── args.py
│   ├── console_graph.py
│   ├── how_far_we_go.py
│   ├── process_obs.py
│   └── show_icon.py
└── main.py
```

## ***KEY NOTE***:

For every unique `environment` (i.e. kitchen, maze2d, mujoco), we will different unqiue ways of handling data input, which will beusing different `train` functions (i.e. train_mixed(), train_batched(), train_split()) within the unique `model` inheritence class (i.e. DtTrainer).
- A specific model's `Trainer` should be generic enough such that when taking `yaml` + `env_name` + `train_type`, it should train.

Similarly, during inference time, an wrapper wouldbe casted upon each environment for working with unqiue model choices. The data_loader should automatically handle environment differecnes during training.

- Currently we still have many `legacy files` that contain single run files, which will be deleted later.

# Contribution Rules
The following is the coding convention of this repository

## 扩展新的辅助类
1. 新建的子类需要继承原有的抽象类
2. 当创造子类父类继承关系时， 如果代码量不大的情况下， 可以考虑放在同一个文件。 但是如果存在的子类超过八个或者代码过长，考虑将其原本的文件变为folder，将各个类放在目录下不同的python文件里，并在目录的__init__.py里面import他们.
    - eg: 目前结构为：agent.trainer.py，使用 from agent.trainer import LptTrainer 调用trainer
3. 当trainer多于八种，代码量超过300行时，项目结构改为专属的文件夹 📁
    - 这样其他文件中的 `from agent.trainer import LptTrainer` 依然有效

    ```bash     
    |- trainer 
        |- __init__.py
        |- LptTrainer.py
        |- PtTrainer.py
                ...
    ```

4. 对于新补充的代码，在/test 加上它的测试，包括它的定义，调用

## 变量定义
1. 如果需要在后续改变的变量：写入yaml的字段中
    - 如果是类似普朗克常量或者Pi之类的不变量， 考虑以常量形式定义在configs/constant.py中
    - 为了测试方便， 仅在tests 文件夹中的测试文件允许magic number


## 新建模型类
当新建模型类时，如果它是一个直接引用和训练的模型（basicPT和LPT），而非模型结构的一部分（例如unet）。 你需要
1. 用@register_model 来为它注册一个名字 假设为name_A
2. 在config文件中添加name_A 所使用的各种参数
3. 为name_A 添加对应的data_loader， 考虑到不同模型也许需要不同的损失函数，optimizer等达到最优

对于新补充的模型，在/test 中加上它的测试，包括它的定义，调用和在toy_dataset上的收敛

***所有的模型是通过子类继承而实现的***

## 新建环境
新建环境需要在data/dataset.py 里面创建新的datasets，并在inferencer.py里面补充make_env的方法

***所有的环境是通过wrapper function来实现的***

## 新建helper_function
如果helper_function 和 helper_class 数量较小，并且只被一个py文件调用，可以放在该文件的开头, 否则放在util里面使用        

## 命名风格
py文件，function, 和变量采用下划线命名风格， 类名采用驼峰命名法（例如：EnvMaker或者 LptTrainer）
