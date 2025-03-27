# 导入必要的库
import os
import argparse
import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

from model import DeepBindNet
from model_gated import DeepBindNetGated
from protein_feature import ProteinFeatureExtractor

# 禁用PyTorch Hub的警告
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepBindNet优化版预测脚本')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'gated'],
                        help='模型类型: standard或gated')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='模型目录，如果为None则使用默认目录')
    
    # 预测模式
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'],
                        help='预测模式: single(单个预测)或batch(批量预测)')
    
    # 单个预测参数
    parser.add_argument('--smiles', type=str, default=None,
                        help='分子SMILES字符串 (单个预测模式)')
    parser.add_argument('--protein_sequence', type=str, default=None,
                        help='蛋白质序列 (单个预测模式)')
    
    # 批量预测参数
    parser.add_argument('--input_csv', type=str, default=None,
                        help='输入CSV文件，包含smiles和protein_sequence列 (批量预测模式)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小 (批量预测模式)')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='预测设备')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化注意力权重 (单个预测模式)')
    parser.add_argument('--output_dir', type=str, default='prediction_results',
                        help='输出目录')
    parser.add_argument('--analyze_averaging', action='store_true',
                        help='是否分析平均化现象 (批量预测模式且有真实值时)')
    
    return parser.parse_args()

def load_model(model_dir, device, model_type='standard'):
    """加载模型"""
    # 设置默认模型目录
    if model_dir is None:
        if model_type == 'gated':
            model_dir = 'outputs_gated'
        else:
            model_dir = 'outputs'
    
    # 加载检查点
    checkpoint_path = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到模型检查点: {checkpoint_path}")
    
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载scaler参数
    scaler_params = checkpoint.get('scaler_params')
    
    # 创建模型
    if model_type == 'gated':
        model = DeepBindNetGated(
            atom_feature_dim=6,
            bond_feature_dim=3,
            hidden_dim=512,  # 匹配训练时的hidden_dim
            feature_dim=256,
            esm_model_path=None,
            fusion_heads=8,
            fusion_layers=5,  # 匹配训练时的fusion_layers
            dropout_rate=0.0,  # 预测时设为0
            scaler_params=scaler_params
        ).to(device)
    else:
        model = DeepBindNet(
            atom_feature_dim=6,
            bond_feature_dim=3,
            hidden_dim=256,
            feature_dim=512,  # 匹配优化版训练脚本的默认值
            esm_model_path=None,
            fusion_heads=8,
            fusion_layers=4,  # 匹配优化版训练脚本的默认值
            dropout_rate=0.0,  # 预测时设为0
            scaler_params=scaler_params
        ).to(device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, scaler_params

def smiles_to_graph(smiles):
    """将SMILES转换为分子图"""
    # 解析SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效的SMILES字符串: {smiles}")
    
    # 添加氢原子
    mol = Chem.AddHs(mol)
    
    # 计算2D坐标
    AllChem.Compute2DCoords(mol)
    
    # 提取原子特征
    num_atoms = mol.GetNumAtoms()
    x = []
    for atom in mol.GetAtoms():
        # 原子特征: 原子类型(one-hot), 形式电荷, 杂化类型, 芳香性, 氢原子数量, 是否在环中
        atom_type = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        hybridization = atom.GetHybridization().real
        is_aromatic = int(atom.GetIsAromatic())
        num_h = atom.GetTotalNumHs()
        is_in_ring = int(atom.IsInRing())
        
        # 简化特征为6维向量
        features = [atom_type, formal_charge, hybridization, is_aromatic, num_h, is_in_ring]
        x.append(features)
    
    x = torch.tensor(x, dtype=torch.float)
    
    # 提取边特征
    edge_indices = []
    edge_attrs = []
    
    for bond in mol.GetBonds():
        # 获取键的起点和终点
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # 键类型
        bond_type = bond.GetBondTypeAsDouble()
        is_conjugated = int(bond.GetIsConjugated())
        is_in_ring = int(bond.IsInRing())
        
        # 边特征: 键类型, 共轭, 是否在环中
        edge_attr = [bond_type, is_conjugated, is_in_ring]
        
        # 添加正向边
        edge_indices.append([i, j])
        edge_attrs.append(edge_attr)
        
        # 添加反向边
        edge_indices.append([j, i])
        edge_attrs.append(edge_attr)
    
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # 处理没有边的情况
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    
    # 创建数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

def predict_single(model, mol_graph, protein_sequence, device):
    """使用模型进行单个预测"""
    model.eval()
    
    with torch.no_grad():
        # 将分子图移动到设备
        mol_graph = mol_graph.to(device)
        
        # 确保模型的protein_encoder也在正确的设备上
        model.protein_encoder = model.protein_encoder.to(device)
        
        # 提取蛋白质特征 (确保在同一设备上)
        protein_features = model.protein_encoder(sequences=[protein_sequence])
        
        # 确保所有张量都在同一设备上
        if protein_features.device != device:
            protein_features = protein_features.to(device)
        
        # 预测
        predictions, fused_features, attn_weights = model(mol_graph, protein_features)
        
        # 返回预测结果和注意力权重
        return predictions.item(), fused_features, attn_weights

def prepare_batch(smiles_list, device):
    """准备分子图批次"""
    mol_graphs = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            mol_graph = smiles_to_graph(smiles)
            mol_graphs.append(mol_graph)
            valid_indices.append(i)
        except Exception as e:
            print(f"警告: 处理SMILES时出错 '{smiles}': {e}")
    
    if not mol_graphs:
        return None, []
    
    # 批处理分子图
    batch = Batch.from_data_list(mol_graphs)
    batch = batch.to(device)
    
    return batch, valid_indices

def batch_predict(model, data_loader, device, output_dir, analyze_averaging=False):
    """批量预测"""
    model.eval()
    results = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始批量预测...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader)):
            smiles_list = batch_data['smiles']
            protein_sequences = batch_data['protein_sequence']
            
            # 准备分子图批次
            mol_graphs_batch, valid_indices = prepare_batch(smiles_list, device)
            
            if mol_graphs_batch is None or len(valid_indices) == 0:
                print(f"批次 {batch_idx+1} 中没有有效的分子")
                continue
            
            # 获取有效的蛋白质序列
            valid_proteins = [protein_sequences[i] for i in valid_indices]
            valid_smiles = [smiles_list[i] for i in valid_indices]
            
            # 获取真实值（如果存在）
            has_targets = 'kiba_scores' in batch_data
            if has_targets:
                valid_targets = [batch_data['kiba_scores'][i] for i in valid_indices]
            
            # 确保模型的protein_encoder也在正确的设备上
            model.protein_encoder = model.protein_encoder.to(device)
            
            # 提取蛋白质特征 (确保在同一设备上)
            protein_features = model.protein_encoder(sequences=valid_proteins)
            
            # 确保所有张量都在同一设备上
            if protein_features.device != device:
                protein_features = protein_features.to(device)
            
            # 预测
            predictions, _, _ = model(mol_graphs_batch, protein_features)
            
            # 处理预测结果
            for i, (smiles, protein_seq, pred) in enumerate(zip(valid_smiles, valid_proteins, predictions)):
                kiba_score = pred.item()
                binding_strength, activity_likelihood = interpret_kiba_score(kiba_score)
                
                result = {
                    'smiles': smiles,
                    'protein_sequence': protein_seq,
                    'kiba_score': kiba_score,
                    'binding_strength': binding_strength,
                    'activity_likelihood': activity_likelihood
                }
                
                # 如果有真实值，添加到结果中
                if has_targets:
                    result['true_kiba_score'] = valid_targets[i]
                
                results.append(result)
    
    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'batch_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    
    # 保存为pickle文件
    pkl_path = os.path.join(output_dir, 'batch_predictions.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"批量预测完成! 共处理 {len(results)} 个有效样本")
    print(f"结果已保存到: {csv_path}")
    
    # 生成结果统计图
    generate_statistics(results_df, output_dir)
    
    # 如果有真实值且需要分析平均化现象，进行分析
    if 'true_kiba_score' in results_df.columns and analyze_averaging:
        analyze_averaging_phenomenon(results_df, output_dir)
    
    return results_df

def calculate_ci(y_true, y_pred):
    """计算一致性指数 (Concordance Index, CI)
    
    CI衡量模型预测值与真实值之间的相对排序一致性。
    CI = 1 表示预测顺序完全一致（完美排序）
    CI = 0.5 表示随机排序（没有排序能力）
    CI = 0 表示排序完全相反（最差情况）
    
    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        
    返回:
        ci: 一致性指数值
    """
    n = len(y_true)
    # 初始化计数器
    concordant = 0
    total_pairs = 0
    
    # 遍历所有可能的样本对
    for i in range(n):
        for j in range(i+1, n):
            if y_true[i] != y_true[j]:  # 只考虑真实值不相等的情况
                total_pairs += 1
                
                # 如果真实值的排序与预测值的排序一致
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant += 1
                # 如果预测值相等，算作0.5个一致对
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
    
    # 如果没有有效的样本对，返回0.5（随机猜测）
    if total_pairs == 0:
        return 0.5
    
    return concordant / total_pairs

def generate_statistics(results_df, output_dir):
    """生成结果统计图"""
    # 1. KIBA分数分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['kiba_score'], bins=30, alpha=0.7)
    plt.axvline(x=results_df['kiba_score'].mean(), color='r', linestyle='--', 
                label=f'平均值: {results_df["kiba_score"].mean():.2f}')
    plt.xlabel('KIBA分数')
    plt.ylabel('频率')
    plt.title('KIBA分数分布')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kiba_score_distribution.png'), dpi=300)
    plt.close()
    
    # 2. 结合亲和力分类饼图
    binding_counts = results_df['binding_strength'].value_counts()
    plt.figure(figsize=(10, 8))
    plt.pie(binding_counts, labels=binding_counts.index, autopct='%1.1f%%', 
            startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('结合亲和力分布')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'binding_strength_distribution.png'), dpi=300)
    plt.close()
    
    # 3. 如果有真实值，生成预测vs真实值散点图
    if 'true_kiba_score' in results_df.columns:
        plt.figure(figsize=(10, 8))
        plt.scatter(results_df['true_kiba_score'], results_df['kiba_score'], alpha=0.5)
        
        # 添加对角线
        min_val = min(results_df['true_kiba_score'].min(), results_df['kiba_score'].min())
        max_val = max(results_df['true_kiba_score'].max(), results_df['kiba_score'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('真实KIBA分数')
        plt.ylabel('预测KIBA分数')
        plt.title('预测值 vs 真实值')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predictions_vs_targets.png'), dpi=300)
        plt.close()
        
        # 计算性能指标
        rmse = np.sqrt(mean_squared_error(results_df['true_kiba_score'], results_df['kiba_score']))
        r2 = r2_score(results_df['true_kiba_score'], results_df['kiba_score'])
        mae = mean_absolute_error(results_df['true_kiba_score'], results_df['kiba_score'])
        ci = calculate_ci(results_df['true_kiba_score'], results_df['kiba_score'])
        
        # 保存性能指标
        metrics = {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'ci': ci
        }
        
        with open(os.path.join(output_dir, 'performance_metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
        
        # 打印性能指标
        print("\n性能指标:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"CI: {ci:.4f} (一致性指数，越接近1表示排序能力越强)")
        
        # 按值域范围分析预测性能
        range_analysis = analyze_predictions_by_range(
            np.array(results_df['kiba_score']),
            np.array(results_df['true_kiba_score']),
            num_bins=5
        )
        
        # 保存分区间分析结果
        range_df = pd.DataFrame.from_dict(range_analysis, orient='index')
        range_df.to_csv(os.path.join(output_dir, 'range_analysis.csv'))
        
        # 打印分区间分析结果
        print("\n按值域范围分析预测性能:")
        for bin_name, metrics in range_analysis.items():
            print(f"{bin_name}: 样本数={metrics['count']} ({metrics['percentage']:.1f}%), "
                  f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
    
    # 4. 统计信息
    stats = {
        '样本数量': len(results_df),
        'KIBA分数平均值': results_df['kiba_score'].mean(),
        'KIBA分数中位数': results_df['kiba_score'].median(),
        'KIBA分数最大值': results_df['kiba_score'].max(),
        'KIBA分数最小值': results_df['kiba_score'].min(),
        'KIBA分数标准差': results_df['kiba_score'].std(),
        '结合亲和力分布': binding_counts.to_dict()
    }
    
    # 保存统计信息
    with open(os.path.join(output_dir, 'statistics.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    # 打印统计信息
    print("\n预测结果统计:")
    print(f"样本数量: {stats['样本数量']}")
    print(f"KIBA分数平均值: {stats['KIBA分数平均值']:.4f}")
    print(f"KIBA分数中位数: {stats['KIBA分数中位数']:.4f}")
    print(f"KIBA分数范围: [{stats['KIBA分数最小值']:.4f}, {stats['KIBA分数最大值']:.4f}]")
    print(f"KIBA分数标准差: {stats['KIBA分数标准差']:.4f}")
    print("\n结合亲和力分布:")
    for strength, count in stats['结合亲和力分布'].items():
        print(f"  {strength}: {count} ({count/stats['样本数量']*100:.1f}%)")

def analyze_predictions_by_range(predictions, targets, num_bins=5):
    """按值域范围分析预测性能"""
    # 计算目标值的分位数
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(targets, quantiles)
    
    # 初始化结果字典
    results = {}
    
    # 对每个区间计算指标
    for i in range(num_bins):
        bin_name = f"Bin {i+1} [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]"
        
        # 获取当前区间的样本
        mask = (targets >= bin_edges[i]) & (targets <= bin_edges[i+1])
        bin_targets = targets[mask]
        bin_preds = predictions[mask]
        
        # 如果区间内有样本，计算指标
        if len(bin_targets) > 0:
            bin_rmse = np.sqrt(mean_squared_error(bin_targets, bin_preds))
            bin_mae = mean_absolute_error(bin_targets, bin_preds)
            bin_r2 = r2_score(bin_targets, bin_preds) if len(bin_targets) > 1 else float('nan')
            bin_ci = calculate_ci(bin_targets, bin_preds) if len(bin_targets) > 1 else float('nan')
            bin_count = len(bin_targets)
            bin_bias = np.mean(bin_preds - bin_targets)
            
            results[bin_name] = {
                'count': bin_count,
                'percentage': bin_count / len(targets) * 100,
                'rmse': bin_rmse,
                'mae': bin_mae,
                'r2': bin_r2,
                'ci': bin_ci,
                'bias': bin_bias
            }
    
    return results

def analyze_averaging_phenomenon(results_df, output_dir):
    """分析预测平均化现象"""
    # 计算目标值与均值的偏差
    target_mean = results_df['true_kiba_score'].mean()
    results_df['target_deviation'] = results_df['true_kiba_score'] - target_mean
    
    # 计算预测偏差
    results_df['prediction_error'] = results_df['kiba_score'] - results_df['true_kiba_score']
    
    # 绘制目标偏差与预测偏差的关系图
    plt.figure(figsize=(10, 8))
    plt.scatter(results_df['target_deviation'], results_df['prediction_error'], alpha=0.5)
    
    # 添加趋势线
    z = np.polyfit(results_df['target_deviation'], results_df['prediction_error'], 1)
    p = np.poly1d(z)
    plt.plot(np.sort(results_df['target_deviation']), p(np.sort(results_df['target_deviation'])), 
             'r--', linewidth=2)
    
    # 添加水平线表示无偏差
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # 添加标签和标题
    plt.xlabel('目标值偏离均值的程度')
    plt.ylabel('预测偏差 (预测值 - 真实值)')
    plt.title('平均化现象分析')
    
    # 添加趋势线方程和平均化系数
    averaging_coefficient = z[0]
    plt.annotate(f'趋势线: y = {z[0]:.4f}x + {z[1]:.4f}\n平均化系数: {averaging_coefficient:.4f}', 
                 xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 添加解释
    plt.figtext(0.5, 0.01, 
                '平均化系数接近-1表示严重的平均化现象\n'
                '平均化系数接近0表示预测偏差与目标值偏离均值无关', 
                ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'averaging_analysis.png'), dpi=300)
    plt.close()
    
    # 保存平均化分析结果
    averaging_analysis = {
        'averaging_coefficient': averaging_coefficient,
        'trend_line_intercept': z[1],
        'correlation': np.corrcoef(results_df['target_deviation'], results_df['prediction_error'])[0, 1]
    }
    
    with open(os.path.join(output_dir, 'averaging_analysis.pkl'), 'wb') as f:
        pickle.dump(averaging_analysis, f)
    
    # 打印平均化分析结果
    print("\n平均化现象分析:")
    print(f"平均化系数: {averaging_coefficient:.4f}")
    print(f"趋势线截距: {z[1]:.4f}")
    print(f"相关系数: {averaging_analysis['correlation']:.4f}")
    
    if averaging_coefficient < -0.8:
        print("结论: 模型存在严重的平均化现象，对极端值的预测能力较弱")
    elif averaging_coefficient < -0.5:
        print("结论: 模型存在中度平均化现象，对极端值的预测有一定偏差")
    elif averaging_coefficient < -0.2:
        print("结论: 模型存在轻微平均化现象，但整体预测较为平衡")
    else:
        print("结论: 模型几乎不存在平均化现象，对极端值的预测能力良好")

def visualize_attention(attn_weights, smiles, protein_sequence, output_dir):
    """可视化注意力权重"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"警告: 无法解析SMILES进行可视化: {smiles}")
        return
    
    # 获取原子符号
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    # 创建蛋白质序列标签 (简化为前10个氨基酸 + ... + 后10个氨基酸)
    if len(protein_sequence) > 20:
        protein_label = f"{protein_sequence[:10]}...{protein_sequence[-10:]}"
    else:
        protein_label = protein_sequence
    
    # 提取最后一层的注意力权重
    last_layer_attn = attn_weights[-1]
    
    # 对所有头的注意力权重取平均
    avg_attn = last_layer_attn.mean(dim=0)
    
    # 可视化注意力权重
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_attn.cpu().numpy(), cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Molecule Atoms')
    plt.ylabel('Protein Features')
    plt.title(f'Attention Weights: {protein_label} -> {smiles}')
    
    # 设置x轴标签为原子符号
    if len(atom_symbols) <= 20:  # 只有当原子数量合理时才显示标签
        plt.xticks(range(len(atom_symbols)), atom_symbols)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_weights.png'), dpi=300)
    plt.close()
    
    print(f"注意力权重可视化已保存到: {os.path.join(output_dir, 'attention_weights.png')}")

def interpret_kiba_score(score):
    """解释分数"""
    if score < 1:
        return "极强结合亲和力", "非常可能有药理活性"
    elif score < 2:
        return "强结合亲和力", "很可能有药理活性"
    elif score < 3:
        return "中等结合亲和力", "可能有药理活性"
    elif score < 4.0:
        return "弱结合亲和力", "药理活性有限"
    else:
        return "极弱结合亲和力", "药理活性可能性低"

class DataLoader:
    """简单的数据加载器"""
    def __init__(self, data_df, batch_size=32):
        self.data = data_df
        self.batch_size = batch_size
        self.n_samples = len(data_df)
        self.indices = list(range(self.n_samples))
    
    def __iter__(self):
        self.current = 0
        return self
    
    def __next__(self):
        if self.current >= self.n_samples:
            raise StopIteration
        
        end_idx = min(self.current + self.batch_size, self.n_samples)
        batch_indices = self.indices[self.current:end_idx]
        batch_data = self.data.iloc[batch_indices]
        
        self.current = end_idx
        
        batch_dict = {
            'smiles': batch_data['smiles'].tolist(),
            'protein_sequence': batch_data['protein_sequence'].tolist()
        }
        
        # 如果有真实值，也添加到批次数据中
        if 'kiba_score' in batch_data.columns:
            batch_dict['kiba_scores'] = batch_data['kiba_score'].tolist()
        
        return batch_dict
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f'使用设备: {device}')
    
    try:
        # 加载模型
        model, scaler_params = load_model(args.model_dir, device, args.model_type)
        
        # 根据模式进行预测
        if args.mode == 'single':
            # 检查必要的参数
            if args.smiles is None or args.protein_sequence is None:
                raise ValueError("单个预测模式需要提供--smiles和--protein_sequence参数")
            
            # 将SMILES转换为分子图
            mol_graph = smiles_to_graph(args.smiles)
            
            # 进行预测
            print(f"\n预测中...")
            print(f"分子SMILES: {args.smiles}")
            print(f"蛋白质序列: {args.protein_sequence[:10]}...{args.protein_sequence[-10:]}")
            
            kiba_score, fused_features, attn_weights = predict_single(model, mol_graph, args.protein_sequence, device)
            
            # 解释KIBA分数
            binding_strength, activity_likelihood = interpret_kiba_score(kiba_score)
            
            # 打印结果
            print("\n预测结果:")
            print(f"KIBA分数: {kiba_score:.4f}")
            print(f"结合亲和力: {binding_strength}")
            print(f"药理活性: {activity_likelihood}")
            
            # 可视化注意力权重
            if args.visualize:
                visualize_attention(attn_weights, args.smiles, args.protein_sequence, args.output_dir)
            
            # 保存结果
            results = {
                'smiles': args.smiles,
                'protein_sequence': args.protein_sequence,
                'kiba_score': kiba_score,
                'binding_strength': binding_strength,
                'activity_likelihood': activity_likelihood,
                'model_type': args.model_type
            }
            
            with open(os.path.join(args.output_dir, 'prediction_result.pkl'), 'wb') as f:
                pickle.dump(results, f)
            
            print(f"\n预测完成! 结果已保存到: {os.path.join(args.output_dir, 'prediction_result.pkl')}")
            
        elif args.mode == 'batch':
            # 检查必要的参数
            if args.input_csv is None:
                raise ValueError("批量预测模式需要提供--input_csv参数")
            
            # 加载输入数据
            if not os.path.exists(args.input_csv):
                raise FileNotFoundError(f"未找到输入文件: {args.input_csv}")
            
            input_df = pd.read_csv(args.input_csv)
            
            # 检查必要的列
            required_columns = ['smiles', 'protein_sequence']
            for col in required_columns:
                if col not in input_df.columns:
                    raise ValueError(f"输入CSV必须包含'{col}'列")
            
            print(f"加载了 {len(input_df)} 个样本")
            
            # 创建数据加载器
            data_loader = DataLoader(input_df, batch_size=args.batch_size)
            
            # 批量预测
            results_df = batch_predict(model, data_loader, device, args.output_dir, args.analyze_averaging)
            
            print(f"\n批量预测完成! 结果已保存到目录: {args.output_dir}")
        
    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
