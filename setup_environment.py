#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepBindNet 环境安装脚本

此脚本自动安装 DeepBindNet 项目所需的所有依赖项。
它会检测系统环境，安装适当版本的 PyTorch、PyTorch Geometric、RDKit 和其他依赖项。

使用方法:
    python setup_environment.py [--cpu-only] [--cuda-version CUDA_VERSION]

参数:
    --cpu-only: 仅安装 CPU 版本的依赖项（不使用 GPU）
    --cuda-version: 指定 CUDA 版本（默认自动检测）
"""

import os
import sys
import subprocess
import platform
import argparse
import pkg_resources
import re

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='安装 DeepBindNet 环境')
    parser.add_argument('--cpu-only', action='store_true', help='仅安装 CPU 版本（不使用 GPU）')
    parser.add_argument('--cuda-version', type=str, help='指定 CUDA 版本（例如 11.8 或 12.1）')
    return parser.parse_args()

def run_command(command, description=None):
    """运行命令并打印输出"""
    if description:
        print(f"\n{description}...")
    
    print(f"执行: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def detect_cuda():
    """检测系统上安装的 CUDA 版本"""
    try:
        # 尝试使用 nvcc 检测 CUDA 版本
        result = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # 解析版本字符串
            version_match = re.search(r'release (\d+\.\d+)', result.stdout)
            if version_match:
                return version_match.group(1)
        
        # 如果 nvcc 不可用，尝试从 nvidia-smi
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            version_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
            if version_match:
                return version_match.group(1)
    except Exception as e:
        print(f"检测 CUDA 版本时出错: {e}")
    
    return None

def get_torch_version():
    """获取已安装的 PyTorch 版本"""
    try:
        import torch
        return torch.__version__.split('+')[0]  # 移除 CUDA 后缀
    except ImportError:
        return None

def install_pytorch(cuda_version=None, cpu_only=False):
    """安装 PyTorch"""
    if cpu_only or cuda_version is None:
        # 安装 CPU 版本
        command = "pip install torch torchvision torchaudio"
    else:
        # 安装 GPU 版本
        # 根据 CUDA 版本选择适当的 PyTorch 版本
        cuda_major = cuda_version.split('.')[0]
        if int(cuda_major) >= 12:
            # CUDA 12.x
            command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        elif int(cuda_major) == 11:
            # CUDA 11.x
            command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        else:
            print(f"警告: CUDA {cuda_version} 可能不受最新 PyTorch 支持，将尝试安装 CPU 版本")
            command = "pip install torch torchvision torchaudio"
    
    return run_command(command, "安装 PyTorch")

def install_pytorch_geometric(torch_version, cuda_version=None, cpu_only=False):
    """安装 PyTorch Geometric 及其依赖项"""
    # 安装主包
    success = run_command("pip install torch-geometric", "安装 PyTorch Geometric")
    if not success:
        return False
    
    # 安装扩展包
    if cpu_only or cuda_version is None:
        ext_url = f"https://data.pyg.org/whl/torch-{torch_version}+cpu.html"
    else:
        # 根据 CUDA 版本选择适当的 URL
        cuda_major = cuda_version.split('.')[0]
        if int(cuda_major) >= 12:
            ext_url = f"https://data.pyg.org/whl/torch-{torch_version}+cu121.html"
        elif int(cuda_major) == 11:
            ext_url = f"https://data.pyg.org/whl/torch-{torch_version}+cu118.html"
        else:
            print(f"警告: CUDA {cuda_version} 可能不受最新 PyTorch Geometric 支持，将尝试安装 CPU 版本")
            ext_url = f"https://data.pyg.org/whl/torch-{torch_version}+cpu.html"
    
    command = f"pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f {ext_url}"
    return run_command(command, "安装 PyTorch Geometric 扩展")

def install_rdkit():
    """安装 RDKit"""
    # 检查是否有 conda 环境
    is_conda = os.environ.get('CONDA_PREFIX') is not None
    
    if is_conda:
        command = "conda install -c conda-forge rdkit -y"
    else:
        command = "pip install rdkit"
    
    return run_command(command, "安装 RDKit")

def install_esm():
    """安装 ESM 模型"""
    return run_command("pip install fair-esm", "安装 ESM 蛋白质语言模型")

def install_requirements():
    """安装其他依赖项"""
    return run_command("pip install -r requirements.txt", "安装其他依赖项")

def verify_installation():
    """验证安装是否成功"""
    try:
        # 尝试导入关键包
        import torch
        import torch_geometric
        import rdkit
        import esm
        import numpy
        import pandas
        
        print("\n验证成功! 所有关键依赖项已正确安装。")
        
        # 检查 CUDA 是否可用
        if torch.cuda.is_available():
            print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 版本: {torch.version.cuda}")
        else:
            print("CUDA 不可用，将使用 CPU 模式。")
        
        return True
    except ImportError as e:
        print(f"\n验证失败: {e}")
        return False

def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("DeepBindNet 环境安装")
    print("=" * 80)
    
    # 检测系统信息
    system = platform.system()
    print(f"操作系统: {system} {platform.version()}")
    
    # 检测 CUDA
    cuda_version = args.cuda_version
    if not args.cpu_only and cuda_version is None:
        cuda_version = detect_cuda()
    
    if args.cpu_only:
        print("将安装 CPU 版本（不使用 GPU）")
    elif cuda_version:
        print(f"检测到 CUDA 版本: {cuda_version}")
    else:
        print("未检测到 CUDA，将安装 CPU 版本")
    
    # 安装 PyTorch
    if not install_pytorch(cuda_version, args.cpu_only):
        print("PyTorch 安装失败，请检查错误信息并手动安装。")
        return
    
    # 获取安装的 PyTorch 版本
    torch_version = get_torch_version()
    if torch_version is None:
        print("无法检测 PyTorch 版本，安装失败。")
        return
    
    print(f"已安装 PyTorch 版本: {torch_version}")
    
    # 安装 PyTorch Geometric
    if not install_pytorch_geometric(torch_version, cuda_version, args.cpu_only):
        print("PyTorch Geometric 安装失败，请检查错误信息并手动安装。")
        return
    
    # 安装 RDKit
    if not install_rdkit():
        print("RDKit 安装失败，请检查错误信息并手动安装。")
        return
    
    # 安装 ESM 模型
    if not install_esm():
        print("ESM 模型安装失败，请检查错误信息并手动安装。")
        return
    
    # 安装其他依赖项
    if not install_requirements():
        print("其他依赖项安装失败，请检查错误信息并手动安装。")
        return
    
    # 验证安装
    if verify_installation():
        print("\n恭喜! DeepBindNet 环境安装成功。")
        print("\n您现在可以运行以下命令来预处理数据:")
        print("  python preprocess.py")
        print("\n然后训练模型:")
        print("  python train.py")
    else:
        print("\n安装可能不完整，请检查错误信息并手动安装缺失的依赖项。")

if __name__ == "__main__":
    main()
