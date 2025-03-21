# 导入必要的库
import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

class DeepBindNetDataset(Dataset):
    """
    DeepBindNet数据集类，用于加载预处理后的蛋白质-小分子相互作用数据
    """
    def __init__(self, data_path):
        """
        初始化数据集
        
        参数:
        - data_path: 预处理数据文件路径
        """
        self.data_path = data_path
        
        # 加载数据
        print(f"加载数据集: {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"数据集加载完成，共 {len(self.data)} 条记录")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取数据集中的一条记录"""
        item = self.data[idx]
        
        # 提取数据
        protein_feature = torch.tensor(item['protein_feature'], dtype=torch.float)
        mol_graph = item['mol_graph']
        kiba_score = torch.tensor(item['kiba_score'], dtype=torch.float)
        
        return {
            'protein_feature': protein_feature,
            'mol_graph': mol_graph,
            'kiba_score': kiba_score,
            'protein_sequence': item['protein_sequence'],
            'smiles': item['smiles']
        }

def collate_fn(batch):
    """
    自定义批处理函数，用于处理不同大小的分子图
    
    参数:
    - batch: 批次数据列表
    
    返回:
    - 批处理后的数据字典
    """
    # 提取数据
    protein_features = [item['protein_feature'] for item in batch]
    mol_graphs = [item['mol_graph'] for item in batch]
    kiba_scores = [item['kiba_score'] for item in batch]
    protein_sequences = [item['protein_sequence'] for item in batch]
    smiles_list = [item['smiles'] for item in batch]
    
    # 将蛋白质特征堆叠为批次
    protein_features_batch = torch.stack(protein_features)
    
    # 将分子图批处理
    mol_graphs_batch = Batch.from_data_list(mol_graphs)
    
    # 将KIBA分数堆叠为批次
    kiba_scores_batch = torch.stack(kiba_scores)
    
    return {
        'protein_features': protein_features_batch,
        'mol_graphs': mol_graphs_batch,
        'kiba_scores': kiba_scores_batch,
        'protein_sequences': protein_sequences,
        'smiles': smiles_list
    }

def get_data_loaders(train_path, val_path, test_path, batch_size=256, num_workers=4):
    """
    创建训练、验证和测试数据加载器
    
    参数:
    - train_path: 训练数据路径
    - val_path: 验证数据路径
    - test_path: 测试数据路径
    - batch_size: 批次大小
    - num_workers: 数据加载线程数
    
    返回:
    - 训练、验证和测试数据加载器
    """
    # 创建数据集
    train_dataset = DeepBindNetDataset(train_path)
    val_dataset = DeepBindNetDataset(val_path)
    test_dataset = DeepBindNetDataset(test_path)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}, 批次数: {len(train_loader)}")
    print(f"验证集大小: {len(val_dataset)}, 批次数: {len(val_loader)}")
    print(f"测试集大小: {len(test_dataset)}, 批次数: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

# 使用示例
if __name__ == "__main__":
    # 设置路径
    data_dir = "data/processed"
    train_path = os.path.join(data_dir, "train_data.pkl")
    val_path = os.path.join(data_dir, "val_data.pkl")
    test_path = os.path.join(data_dir, "test_data.pkl")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        train_path, val_path, test_path, batch_size=32
    )
    
    # 测试数据加载
    for batch_idx, batch in enumerate(train_loader):
        print(f"批次 {batch_idx+1}:")
        print(f"蛋白质特征形状: {batch['protein_features'].shape}")
        print(f"分子图节点数: {batch['mol_graphs'].x.shape[0]}")
        print(f"分子图边数: {batch['mol_graphs'].edge_index.shape[1]}")
        print(f"KIBA分数形状: {batch['kiba_scores'].shape}")
        print(f"批次中的分子数: {len(batch['smiles'])}")
        
        # 只打印第一个批次
        break
