import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
from torch_geometric.data import Data, Batch
from model import DeepBindNet
from model_gated import DeepBindNetGated

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 优先使用中文黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 检查是否有可用的中文字体
try:
    font_names = [f.name for f in mpl.font_manager.fontManager.ttflist]
    chinese_fonts = [f for f in font_names if 'SimHei' in f or 'Microsoft YaHei' in f or 'SimSun' in f or 'FangSong' in f]
    
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = [chinese_fonts[0]] + plt.rcParams['font.sans-serif']
    else:
        print("警告: 未找到中文字体，将使用英文标题")
        USE_ENGLISH = True
except:
    USE_ENGLISH = True
else:
    USE_ENGLISH = False

def create_test_data(batch_size=16, noise_level=0.0):
    """创建测试数据"""
    # 创建分子图数据
    data_list = []
    
    for i in range(batch_size):
        # 随机节点数 (3-10)
        num_nodes = np.random.randint(3, 11)
        # 随机边数 (节点数 * (1-3))
        num_edges = np.random.randint(num_nodes, num_nodes * 3 + 1)
        
        # 创建节点特征
        x = torch.randn(num_nodes, 6)
        
        # 创建边索引 (确保边的数量正确且有效)
        edge_index = torch.zeros((2, num_edges), dtype=torch.long)
        for j in range(num_edges):
            # 随机选择源节点和目标节点
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            # 确保不是自环
            while dst == src:
                dst = np.random.randint(0, num_nodes)
            edge_index[0, j] = src
            edge_index[1, j] = dst
        
        # 创建边特征
        edge_attr = torch.randn(num_edges, 3)
        
        # 添加噪声 (如果指定)
        if noise_level > 0:
            x = x + torch.randn_like(x) * noise_level
            edge_attr = edge_attr + torch.randn_like(edge_attr) * noise_level
        
        # 创建数据对象
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)
    
    # 批处理
    batch = Batch.from_data_list(data_list)
    
    # 创建蛋白质特征
    protein_features = torch.randn(batch_size, 256)
    if noise_level > 0:
        protein_features = protein_features + torch.randn_like(protein_features) * noise_level
    
    return batch, protein_features

def compare_models():
    """比较标准模型和门控模型的性能"""
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建模型
    standard_model = DeepBindNet(
        atom_feature_dim=6,
        bond_feature_dim=3,
        hidden_dim=128,
        feature_dim=256,
        esm_model_path=None,
        fusion_heads=8,
        fusion_layers=2,
        dropout_rate=0.2
    )
    
    gated_model = DeepBindNetGated(
        atom_feature_dim=6,
        bond_feature_dim=3,
        hidden_dim=128,
        feature_dim=256,
        esm_model_path=None,
        fusion_heads=8,
        fusion_layers=2,
        dropout_rate=0.2
    )
    
    # 设置为评估模式
    standard_model.eval()
    gated_model.eval()
    
    # 测试不同噪声级别
    noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0]
    batch_size = 16
    
    # 存储结果
    std_times = []
    gated_times = []
    std_preds = []
    gated_preds = []
    
    for noise in noise_levels:
        print(f"\n=== 噪声级别: {noise:.1f} ===")
        
        # 创建测试数据
        mol_graphs, protein_features = create_test_data(batch_size, noise)
        
        # 测试标准模型
        start_time = time.time()
        with torch.no_grad():
            std_pred, _, _ = standard_model(mol_graphs, protein_features)
        std_time = time.time() - start_time
        std_times.append(std_time)
        std_preds.append(std_pred.detach().numpy())
        
        # 测试门控模型
        start_time = time.time()
        with torch.no_grad():
            gated_pred, _, _ = gated_model(mol_graphs, protein_features)
        gated_time = time.time() - start_time
        gated_times.append(gated_time)
        gated_preds.append(gated_pred.detach().numpy())
        
        # 计算预测差异
        pred_diff = torch.abs(std_pred - gated_pred).mean().item()
        
        print(f"标准模型运行时间: {std_time:.4f}秒")
        print(f"门控模型运行时间: {gated_time:.4f}秒")
        print(f"预测差异 (平均绝对差): {pred_diff:.4f}")
    
    # 绘制运行时间比较
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(noise_levels, std_times, 'o-', label='Standard Model' if USE_ENGLISH else '标准模型')
    plt.plot(noise_levels, gated_times, 's-', label='Gated Model' if USE_ENGLISH else '门控模型')
    plt.xlabel('Noise Level' if USE_ENGLISH else '噪声级别')
    plt.ylabel('Runtime (s)' if USE_ENGLISH else '运行时间 (秒)')
    plt.title('Model Runtime Comparison' if USE_ENGLISH else '模型运行时间比较')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制预测稳定性比较
    plt.subplot(1, 2, 2)
    
    # 计算每个噪声级别下预测的标准差
    std_stds = [np.std(pred) for pred in std_preds]
    gated_stds = [np.std(pred) for pred in gated_preds]
    
    plt.plot(noise_levels, std_stds, 'o-', label='Standard Model' if USE_ENGLISH else '标准模型')
    plt.plot(noise_levels, gated_stds, 's-', label='Gated Model' if USE_ENGLISH else '门控模型')
    plt.xlabel('Noise Level' if USE_ENGLISH else '噪声级别')
    plt.ylabel('Prediction Std Dev' if USE_ENGLISH else '预测标准差')
    plt.title('Prediction Stability Comparison' if USE_ENGLISH else '预测稳定性比较')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()
    
    print("\n模型比较图已保存为 'model_comparison.png'")
    
    # 提取并分析门控系数
    print("\n=== 门控系数分析 ===")
    
    # 获取门控模型的融合模块
    fusion_module = gated_model.fusion_module
    
    # 创建不同噪声级别的测试数据
    gate_values = []
    
    for noise in noise_levels:
        _, protein_features = create_test_data(batch_size, 0.0)  # 蛋白质特征保持不变
        _, noisy_molecule_features = create_test_data(batch_size, noise)  # 分子特征添加噪声
        
        # 提取分子特征
        molecule_features = torch.randn(batch_size, 256)
        noisy_molecule = molecule_features + torch.randn_like(molecule_features) * noise
        
        # 计算门控系数
        with torch.no_grad():
            gate_input = torch.cat([protein_features, noisy_molecule], dim=1)
            g = fusion_module.gate_net(gate_input)
            gate_values.append(g.mean().item())
    
    # 绘制门控系数与噪声级别的关系
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, gate_values, 'o-', linewidth=2)
    plt.xlabel('Noise Level' if USE_ENGLISH else '噪声级别')
    plt.ylabel('Average Gate Coefficient' if USE_ENGLISH else '平均门控系数')
    plt.title('Relationship Between Noise Level and Gate Coefficient' if USE_ENGLISH else '噪声级别与门控系数的关系')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('gate_vs_noise.png', dpi=300)
    plt.close()
    
    print("门控系数分析图已保存为 'gate_vs_noise.png'")
    print(f"门控系数随噪声变化: {gate_values}")

if __name__ == "__main__":
    compare_models()
