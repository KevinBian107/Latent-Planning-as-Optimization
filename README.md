## LPT Training

### Installation

Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install MuJoCo.
Then, dependencies can be installed with the following command:

```bash
conda env create -f env.yml
```

### Downloading datasets

```bash
cd data
python process_data.py
```

## Example usage

Experiments can be reproduced with the following:

```bash
cd scripts
python train.py
```

To customize or experiment with different environments, the environment-specific configurations, such as the number of layers, context length, and learning rate, can be found in `scripts/config.py`. This file defines parameters for each supported environment and allows the script to adjust its training configuration dynamically based on the `--env_name` argument. 

The `config.py` file includes various environments including Gym-Mujoco, Maze2D, and Franka Kitchen environments, with detailed specifications for parameters:
- `n_layer`: Number of layers in the model.
- `n_head`: Number of attention heads.
- `hidden_size`: Dimensionality of the hidden layer.
- `context_len`: Length of the context window for processing input sequences.
- `learning_rate`: Learning rate used during training.
- `langevin_step_size`: Step size for Langevin dynamics.
- `env_targets`: Target return values for each environment.
- `max_len`: Maximum episode length.


## Acknowledgements

The decision transformer model implementation is based on the [official repository of DT](https://github.com/kzl/decision-transformer) .
The latent plan transformer model implementation is based on the [official repository of LPT](https://github.com/mingluzhao/Latent-Plan-Transformer) .
