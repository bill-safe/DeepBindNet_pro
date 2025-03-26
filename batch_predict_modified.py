import os
import argparse
import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data, Batch

from model import DeepBindNet
from model_gated import DeepBindNetGated
from predict import load_model, smiles_to_graph, interpret_kiba_score

# 禁用PyTorch Hub的警告
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepBindNet批量预测脚本')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='gated', choices=['standard', 'gated'],
                        help='模型类型: standard或gated')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='模型目录，如果为None则使用默认目录')
    
    # 输入参数
    parser.add_argument('--input_csv', type=str, required=True,
                        help='输入CSV文件，包含smiles和protein_sequence列')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='预测设备')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--output_dir', type=str, default='batch_prediction_results',
                        help='输出目录')
    
    return parser.parse_args()

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

def batch_predict(model, data_loader, device, output_dir):
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
                
                results.append({
                    'smiles': smiles,
                    'protein_sequence': protein_seq,
                    'kiba_score': kiba_score,
                    'binding_strength': binding_strength,
                    'activity_likelihood': activity_likelihood
                })
    
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
    
    return results_df

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
    
    # 3. 统计信息
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
        
        return {
            'smiles': batch_data['smiles'].tolist(),
            'protein_sequence': batch_data['protein_sequence'].tolist()
        }
    
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
        
        # 加载模型
        model = load_model(args.model_dir, device, args.model_type)
        
        # 创建数据加载器
        data_loader = DataLoader(input_df, batch_size=args.batch_size)
        
        # 批量预测
        results_df = batch_predict(model, data_loader, device, args.output_dir)
        
        print(f"\n批量预测完成! 结果已保存到目录: {args.output_dir}")
        
    except Exception as e:
        print(f"批量预测过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
