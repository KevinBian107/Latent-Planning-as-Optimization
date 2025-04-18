# base image with miniconda
FROM continuumio/miniconda3:latest

# system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.mujoco

# Download and install MuJoCo
# for linux
ARG MUJOCO_URL="https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
# for mac
# ARG MUJOCO_URL="https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz"

RUN wget -q ${MUJOCO_URL} -O /tmp/mujoco.tar.gz && \
    tar -xzf /tmp/mujoco.tar.gz -C /root/.mujoco/ && \
    mv /root/.mujoco/MuJoCo* /root/.mujoco/mujoco210 || true && \
    rm /tmp/mujoco.tar.gz


# Set environment variables
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210

# Verify installation
RUN ls /root/.mujoco/mujoco210 && \
    test -f /root/.mujoco/mujoco210/bin/simulate

# Python dependencies
COPY env.yml .
RUN conda env create -f env.yml && \
    echo "conda activate $(head -1 env.yml | cut -d' ' -f2)" >> ~/.bashrc

# Set working environment
SHELL ["/bin/bash", "--login", "-c"]

