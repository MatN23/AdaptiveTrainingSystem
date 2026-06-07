# 

:

- PyTorch >= 1.11  PyTorch <= 2.1
- Python >= 3.7
- CUDA >= 11.0
- [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 7.0 (V100/RTX20 and higher)
- Linux OS

 [](https://github.com/hpcaitech/ColossalAI/issues/new/choose)

## PyPI

PyPIColossal-AI

```shell
pip install colossalai
```

**Linux**

PyTorch`BUILD_EXT=1`PyTorch

```shell
BUILD_EXT=1 pip install colossalai
```

## 

>  issue

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install dependency
pip install -r requirements/requirements.txt

# install colossalai
BUILD_EXT=1 pip install .
```

 CUDA `BUILD_EXT=1`

```shell
pip install .
```

CUDA 10.2ColossalAIcub

```bash
# clone the repository
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# download the cub library
wget https://github.com/NVIDIA/cub/archive/refs/tags/1.8.0.zip
unzip 1.8.0.zip
cp -r cub-1.8.0/cub/ colossalai/kernel/cuda_native/csrc/kernels/include/

# install
BUILD_EXT=1 pip install .
```

<!-- doc-test-command: echo "installation.md does not need test" -->
