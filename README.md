# Meta Planning as Inference
...

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
