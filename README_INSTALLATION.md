# DeepBindNet 安装指南

本目录包含了安装和设置 DeepBindNet 项目环境所需的所有文件。以下是对各个文件的简要说明和使用方法。

## 文件概述

- **requirements.txt**: 列出了项目所需的所有 Python 依赖项及其版本要求
- **INSTALLATION.md**: 提供了详细的手动安装步骤和说明
- **DEPENDENCIES.md**: 详细解释了每个依赖项在项目中的具体作用
- **setup_environment.py**: 自动化安装脚本，可以帮助您快速设置环境

## 快速开始

如果您想快速设置环境，可以使用自动化安装脚本：

```bash
# 使用默认设置（自动检测 CUDA）
python setup_environment.py

# 仅使用 CPU 版本
python setup_environment.py --cpu-only

# 指定 CUDA 版本
python setup_environment.py --cuda-version 11.3
```

## 手动安装

如果您更喜欢手动安装或自动化脚本遇到问题，请按照以下步骤操作：

1. 创建并激活虚拟环境（推荐）
2. 安装 PyTorch（根据您的 CUDA 版本选择适当的命令）
3. 安装 PyTorch Geometric 及其扩展
4. 安装 RDKit
5. 安装其他依赖项

详细的手动安装步骤请参考 [INSTALLATION.md](./INSTALLATION.md)。

## 依赖项说明

如果您想了解项目中使用的各个依赖项及其作用，请参考 [DEPENDENCIES.md](./DEPENDENCIES.md)。该文档详细解释了：

- 每个依赖项的版本要求
- 依赖项在项目中的具体应用
- 依赖关系图
- 安装优先级
- 版本兼容性注意事项
- 内存和计算要求

## 安装验证

安装完成后，您可以运行以下命令验证环境是否正确设置：

```bash
python -c "import torch; import torch_geometric; import rdkit; import esm; import numpy; import pandas; print('安装成功！')"
```

如果没有错误消息，则表示安装成功。

## 常见问题解决

### PyTorch Geometric 安装问题

如果在安装 PyTorch Geometric 时遇到问题，请确保：

1. PyTorch 和 CUDA 版本兼容
2. 使用正确的 URL 安装扩展包
3. Windows 用户可能需要安装 Microsoft Visual C++ Build Tools

### RDKit 安装问题

RDKit 在某些平台上可能难以通过 pip 安装。如果遇到问题，建议使用 conda：

```bash
conda install -c conda-forge rdkit
```

### CUDA 相关问题

如果遇到 CUDA 相关错误，请检查：

1. NVIDIA 驱动程序是否正确安装
2. CUDA 工具包版本是否与 PyTorch 兼容
3. 环境变量是否正确设置

## 后续步骤

成功安装环境后，您可以：

1. 准备数据集
2. 运行预处理脚本：`python preprocess.py`
3. 训练模型：`python train.py`

有关更多详细信息，请参阅项目的主要文档。
