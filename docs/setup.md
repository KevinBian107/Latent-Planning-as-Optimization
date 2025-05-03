# Installation

## Conda Setup Guide

1. **Clone the Repository**

```bash
git clone https://github.com/KevinBian107/Meta-Planning-as-Inference.git
cd Meta-Planning-as-Inference
```

2. **Install MuJoCo**

- Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
- Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

You can run the following commands to extract the `mujoco210` directory into `~/.mujoco/mujoco210`.

```bash
mkdir -p ~/.mujoco
mv ~/Downloads/mujoco210 ~/.mujoco/
```

Run the following command to ensure success installation:

```bash
ls ~/.mujoco/mujoco210
```

3. **Install Dependencies**

We recommend using conda to avoid dependencies issues.

```bash
conda env create -f env.yaml
```

or

```bash
conda create -n mpi python=3.8
pip install -r requirements.txt
```

After successful installation, activate your environment:

```bash
conda activate mpi
```

4. **Testing**

You can check if your installation is correct by running the following test files for model training:

```bash
python tests/test_mpill_kitchen_seq.py
```



## Docker Setup Guide

1. **Install Docker, Docker Compose, and NVIDIA Container Toolkit** (Skip this step if already installed):

First check if docker is installed:
```bash
docker --version
```

Install Docker:
```bash
sudo apt update
sudo apt install docker.io
```

Install Docker Compose:
```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.35.0/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

Install NVIDIA Container Toolkit (for CUDA GPU support):
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU is accessible in Docker:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

2. **Clone the Repository**

```bash
git clone https://github.com/KevinBian107/Meta-Planning-as-Inference.git
cd Meta-Planning-as-Inference
```

3. **Build Docker Image**
```bash
docker-compose build
```

4. **Start the Container**
```bash
docker-compose up
```

5. **Debug in a One-off Container** (optional):

You can debug in a one-off container using interactive bash shell:
```bash
docker compose run --service-ports mpi bash
```

