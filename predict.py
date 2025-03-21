import os
import argparse
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from model import DeepBindNet
from model_gated import DeepBindNetGated
from protein_feature import ProteinFeatureExtractor

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepBindNet预测脚本')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='gated', choices=['standard', 'gated'],
                        help='模型类型: standard或gated')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='模型目录，如果为None则使用默认目录')
    
    # 输入参数
    parser.add_argument('--smiles', type=str, required=True,
                        help='分子SMILES字符串')
    parser.add_argument('--protein_sequence', type=str, required=True,
                        help='蛋白质序列')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='预测设备')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化注意力权重')
    parser.add_argument('--output_dir', type=str, default='prediction_results',
                        help='输出目录')
    
    return parser.parse_args()

def load_model(model_dir, device, model_type='gated'):
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
            feature_dim=256,
            esm_model_path=None,
            fusion_heads=8,
            fusion_layers=2,
            dropout_rate=0.0,  # 预测时设为0
            scaler_params=scaler_params
        ).to(device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

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

def predict(model, mol_graph, protein_sequence, device):
    """使用模型进行预测"""
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
    """解释KIBA分数"""
    if score > 12.0:
        return "极强结合亲和力", "非常可能有药理活性"
    elif score > 8.0:
        return "强结合亲和力", "很可能有药理活性"
    elif score > 6.0:
        return "中等结合亲和力", "可能有药理活性"
    elif score > 4.0:
        return "弱结合亲和力", "药理活性有限"
    else:
        return "极弱结合亲和力", "药理活性可能性低"

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
        model = load_model(args.model_dir, device, args.model_type)
        
        # 将SMILES转换为分子图
        mol_graph = smiles_to_graph(args.smiles)
        
        # 进行预测
        print(f"\n预测中...")
        print(f"分子SMILES: {args.smiles}")
        print(f"蛋白质序列: {args.protein_sequence[:10]}...{args.protein_sequence[-10:]}")
        
        kiba_score, fused_features, attn_weights = predict(model, mol_graph, args.protein_sequence, device)
        
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
        
    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
