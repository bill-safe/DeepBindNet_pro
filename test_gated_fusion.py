import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from fusion import CrossAttentionFusion, GatedCrossAttention

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 优先使用中文黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def test_fusion_modules():
    """
    测试并比较标准跨模态注意力和门控跨模态注意力
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(54243)
    np.random.seed(54243)
    
    # 创建融合模块
    standard_fusion = CrossAttentionFusion(
        embed_dim=256,
        num_heads=8,
        ff_dim=1024,
        num_layers=2,
        dropout=0.1
    )
    
    gated_fusion = GatedCrossAttention(
        embed_dim=256,
        num_heads=8,
        ff_dim=1024,
        num_layers=2,
        dropout=0.1
    )
    
    # 设置为评估模式
    standard_fusion.eval()
    gated_fusion.eval()
    
    # 创建测试数据
    batch_size = 10
    protein_features = torch.randn(batch_size, 256)
    molecule_features = torch.randn(batch_size, 256)
    
    # 创建噪声数据（模拟低质量特征）
    noisy_molecule_features = molecule_features.clone() + torch.randn(batch_size, 256) * 2.0
    
    print("=== 测试标准数据 ===")
    # 使用标准数据测试
    with torch.no_grad():
        standard_output, _ = standard_fusion(protein_features, molecule_features)
        gated_output, _ = gated_fusion(protein_features, molecule_features)
        
        # 计算与原始特征的相似度
        standard_similarity = cosine_similarity(standard_output, protein_features)
        gated_similarity = cosine_similarity(gated_output, protein_features)
        
        print(f"标准融合 - 与蛋白质特征的平均余弦相似度: {standard_similarity.mean().item():.4f}")
        print(f"门控融合 - 与蛋白质特征的平均余弦相似度: {gated_similarity.mean().item():.4f}")
    
    print("\n=== 测试噪声数据 ===")
    # 使用噪声数据测试
    with torch.no_grad():
        standard_output_noisy, _ = standard_fusion(protein_features, noisy_molecule_features)
        gated_output_noisy, _ = gated_fusion(protein_features, noisy_molecule_features)
        
        # 计算与原始特征的相似度
        standard_similarity_noisy = cosine_similarity(standard_output_noisy, protein_features)
        gated_similarity_noisy = cosine_similarity(gated_output_noisy, protein_features)
        
        print(f"标准融合 - 与蛋白质特征的平均余弦相似度: {standard_similarity_noisy.mean().item():.4f}")
        print(f"门控融合 - 与蛋白质特征的平均余弦相似度: {gated_similarity_noisy.mean().item():.4f}")
    
    # 提取门控系数
    with torch.no_grad():
        # 获取门控网络的输入
        gate_input = torch.cat([protein_features, molecule_features], dim=1)
        gate_input_noisy = torch.cat([protein_features, noisy_molecule_features], dim=1)
        
        # 手动计算门控系数
        g_normal = gated_fusion.gate_net(gate_input)
        g_noisy = gated_fusion.gate_net(gate_input_noisy)
        
        print(f"\n标准数据的平均门控系数: {g_normal.mean().item():.4f}")
        print(f"噪声数据的平均门控系数: {g_noisy.mean().item():.4f}")
        
        # 绘制门控系数分布
        plot_gate_coefficients(g_normal.detach().numpy(), g_noisy.detach().numpy())

def cosine_similarity(a, b):
    """计算两组向量之间的余弦相似度"""
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    return (a_norm * b_norm).sum(dim=1)

def plot_gate_coefficients(g_normal, g_noisy):
    """绘制门控系数分布"""
    plt.figure(figsize=(10, 6))
    
    # 尝试设置中文字体
    try:
        # 检查是否有可用的中文字体
        font_names = [f.name for f in mpl.font_manager.fontManager.ttflist]
        chinese_fonts = [f for f in font_names if 'SimHei' in f or 'Microsoft YaHei' in f or 'SimSun' in f or 'FangSong' in f]
        
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = [chinese_fonts[0]] + plt.rcParams['font.sans-serif']
        else:
            # 如果没有中文字体，使用英文标题
            print("警告: 未找到中文字体，将使用英文标题")
            use_english = True
    except:
        use_english = True
    
    # 标准数据分布
    plt.subplot(1, 2, 1)
    plt.hist(g_normal, bins=20, alpha=0.7, color='blue')
    plt.axvline(x=g_normal.mean(), color='red', linestyle='--', label=f'Mean: {g_normal.mean():.4f}')
    plt.title('Gate Coefficients - Standard Data')
    plt.xlabel('Gate Coefficient Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 噪声数据分布
    plt.subplot(1, 2, 2)
    plt.hist(g_noisy, bins=20, alpha=0.7, color='orange')
    plt.axvline(x=g_noisy.mean(), color='red', linestyle='--', label=f'Mean: {g_noisy.mean():.4f}')
    plt.title('Gate Coefficients - Noisy Data')
    plt.xlabel('Gate Coefficient Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('gate_coefficients.png', dpi=300)
    plt.close()
    
    print("门控系数分布图已保存为 'gate_coefficients.png'")

if __name__ == "__main__":
    test_fusion_modules()
