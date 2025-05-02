# Meta Planning as Inference
This repository is designed to tackle **continual learning** problems with **long horizon planning**. We try to leverages probabilistic inference along with meta-learning to build offline learning algorithms capable of learning multipl tasks and creating its own diverse strategies, achieving the so-called "learning to learn".

### Installation

Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install MuJoCo. This need tobe manually putted into the system file by:

```bash
mkdir -p ~/.mujoco
mv ~/Downloads/mujoco210 ~/.mujoco/
```

Use the following to ensure success installation:

```bash
ls ~/.mujoco/mujoco210
```

Then, dependencies can be installed with the following command using `conda` or `pip`:

```bash
conda env create env.yaml
pip instal requirements.txt
```

## Acknowledgements

The decision transformer model implementation advices code from the [official repository of DT](https://github.com/kzl/decision-transformer) .

The latent plan transformer model implementation advices code from the [official repository of LPT](https://github.com/mingluzhao/Latent-Plan-Transformer) .
