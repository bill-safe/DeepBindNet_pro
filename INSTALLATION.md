# DeepBindNet 安装指南

本文档提供了安装 DeepBindNet 项目所需的所有依赖项的详细说明。

## 环境要求

- Python 3.8 或更高版本
- CUDA 11.3 或更高版本（用于 GPU 加速，推荐）
- 至少 8GB RAM
- 至少 10GB 磁盘空间（用于模型和数据）

## 安装步骤

### 1. 创建虚拟环境（推荐）

```bash
# 使用 conda 创建虚拟环境
conda create -n deepbindnet python=3.9
conda activate deepbindnet

# 或者使用 venv
python -m venv deepbindnet_env
# 在 Windows 上激活
deepbindnet_env\Scripts\activate
# 在 Linux/Mac 上激活
source deepbindnet_env/bin/activate
```

### 2. 安装 PyTorch

根据您的 CUDA 版本安装适当版本的 PyTorch。以下命令适用于 CUDA 11.3：

```bash
# 使用 conda 安装 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia

# 或者使用 pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
```

如果您没有 GPU，可以安装 CPU 版本：

```bash
pip install torch torchvision torchaudio
```

### 3. 安装 PyTorch Geometric

PyTorch Geometric 是处理图数据的库，安装可能比较复杂，需要与 PyTorch 和 CUDA 版本匹配：

```bash
# 使用 pip 安装 PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html
```

请将 `{TORCH_VERSION}` 和 `{CUDA_VERSION}` 替换为您的 PyTorch 和 CUDA 版本。例如，对于 PyTorch 1.12.0 和 CUDA 11.3：

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```

### 4. 安装 RDKit

RDKit 是用于分子处理的化学信息学工具包：

```bash
# 使用 conda 安装 RDKit（推荐）
conda install -c conda-forge rdkit

# 或者使用 pip
pip install rdkit
```

### 5. 安装 ESM

ESM 是 Facebook 开发的蛋白质语言模型：

```bash
pip install fair-esm
```

### 6. 安装其他依赖项

使用提供的 requirements.txt 文件安装其余依赖项：

```bash
pip install -r requirements.txt
```

## 验证安装

安装完成后，您可以运行以下命令验证安装是否成功：

```bash
python -c "import torch; import torch_geometric; import rdkit; import esm; import numpy; import pandas; print('安装成功！')"
```

如果没有错误消息，则表示安装成功。

## 可能的问题和解决方案

### PyTorch Geometric 安装问题

如果在安装 PyTorch Geometric 时遇到问题，可以尝试以下解决方案：

1. 确保 PyTorch 和 CUDA 版本兼容
2. 对于 Windows 用户，可能需要安装 Microsoft Visual C++ Build Tools
3. 尝试使用预编译的二进制文件：

```bash
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+cpu.html
```

### RDKit 安装问题

如果使用 pip 安装 RDKit 时遇到问题，强烈建议使用 conda 安装：

```bash
conda install -c conda-forge rdkit
```

### CUDA 相关问题

如果遇到 CUDA 相关错误，请确保：

1. 已安装兼容的 NVIDIA 驱动程序
2. CUDA 工具包版本与 PyTorch 兼容
3. 环境变量正确设置

## 数据准备

安装完依赖项后，您需要准备数据集：

1. 下载 KIBA 数据集
2. 将数据集放在 `data/` 目录下
3. 运行预处理脚本：

```bash
python preprocess.py
```

## 模型训练

准备好数据后，您可以开始训练模型：

```bash
python train.py
```

有关更多详细信息，请参阅项目文档。
