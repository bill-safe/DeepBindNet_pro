# 导入必要的库
from rdkit import Chem  # 用于处理化学分子
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算
from torch_geometric.data import Data  # 图数据处理
import torch_scatter  # 图神经网络中的聚合操作

# 分子图处理类，负责将SMILES字符串转换为图数据
class MolecularGraphProcessor:
    def __init__(self):
        # 初始化原子和键的特征维度
        self.atom_feature_dim = 6  # 每个原子的特征维度
        self.bond_feature_dim = 3  # 每条键的特征维度

    # 获取原子特征
    @staticmethod
    def get_atom_features(atom):
        """
        提取原子特征，包括：
        - 原子序数
        - 原子度数
        - 形式电荷
        - 隐式价态
        - 芳香性
        - 杂化状态
        """
        return [
            atom.GetAtomicNum(),           # 原子序数
            atom.GetDegree(),              # 原子度数
            atom.GetFormalCharge(),        # 形式电荷
            atom.GetImplicitValence(),     # 隐式价态
            int(atom.GetIsAromatic()),     # 芳香性
            atom.GetHybridization().real,  # 杂化状态
        ]

    # 获取键特征
    @staticmethod
    def get_bond_features(bond):
        """
        提取键特征，包括：
        - 键类型
        - 共轭性
        - 是否在环上
        """
        return [
            bond.GetBondTypeAsDouble(),    # 单键:1，双键:2，三键:3，芳香键:1.5
            int(bond.GetIsConjugated()),   # 共轭性
            int(bond.IsInRing()),         # 是否处于环上
        ]

    # 处理分子SMILES字符串
    def process_molecule(self, smiles):
        """
        将SMILES字符串转换为图数据
        返回包含节点特征、边索引和边特征的Data对象
        """
        # 将SMILES转换为分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # 获取所有原子的特征
        node_features = [self.get_atom_features(atom) for atom in mol.GetAtoms()]
        edge_index = []  # 存储边的连接关系
        edge_attr = []   # 存储边的特征

        # 遍历所有键
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_feature = self.get_bond_features(bond)
            # 双向添加边（因为图是无向的）
            edge_index += [[start, end], [end, start]]
            edge_attr += [edge_feature, edge_feature]

        # 将数据转换为PyTorch张量
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        node_features = torch.tensor(node_features, dtype=torch.float)

        # 返回图数据对象
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# 残差GIN层实现
class ResidualGINLayer(torch.nn.Module):
    """
    带有残差连接的GIN层，包含：
    - GIN卷积层
    - 残差连接
    - BatchNorm1d防止梯度爆炸
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # GIN层
        self.gin = GINLayer(in_dim, out_dim)
        
        # 如果输入输出维度不同，添加投影层
        self.projection = None
        if in_dim != out_dim:
            self.projection = torch.nn.Linear(in_dim, out_dim)
        
        # BatchNorm1d防止梯度爆炸
        self.bn = torch.nn.BatchNorm1d(out_dim)
    
    def forward(self, x, edge_index, edge_attr):
        """前向传播"""
        # 保存输入用于残差连接
        identity = x
        
        # 通过GIN层
        out = self.gin(x, edge_index, edge_attr)
        
        # 应用残差连接
        if self.projection is not None:
            identity = self.projection(identity)
        out = out + identity
        
        # 应用BatchNorm1d
        # 由于x的形状是[num_nodes, out_dim]，需要处理批次维度
        # 这里假设节点已经按照batch索引排序
        out = self.bn(out)
        
        return out

# GIN图神经网络类
class GIN(torch.nn.Module):
    """
    图神经网络模块，包含：
    - 节点/边特征嵌入层
    - 5层残差GIN卷积，每层后接Dropout
    - 全局池化层（输出384维特征）
    - 输出层
    """
    def __init__(self, atom_feature_dim, bond_feature_dim, hidden_dim=128, out_dim=384, dropout_rate=0.2):
        super().__init__()
        # 节点特征嵌入层
        self.node_embedding = torch.nn.Sequential(
            torch.nn.Linear(atom_feature_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU()
        )
        
        # 边特征嵌入层
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.Linear(bond_feature_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU()
        )
        
        # 5层残差GIN卷积（减少层数，添加残差连接）
        self.gin_layers = torch.nn.ModuleList([
            ResidualGINLayer(hidden_dim, hidden_dim) for _ in range(5)
        ])
        
        # Dropout层
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # 全局池化层 - 使用全局平均池化确保输出固定维度
        # 不能将scatter_mean放入Sequential中，因为它不是Module子类
        self.global_pool_linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, out_dim)
        )
        
        # 输出层
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(out_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
            torch.nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        """前向传播过程"""
        # 如果没有提供batch信息，则假设所有节点属于同一个图
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 节点特征嵌入
        x = self.node_embedding(x)
        
        # 边特征嵌入
        edge_attr = self.edge_embedding(edge_attr)
        
        # 存储所有层的输出用于层间连接
        layer_outputs = []
        
        # 通过残差GIN层
        for layer in self.gin_layers:
            x = layer(x, edge_index, edge_attr)
            x = torch.nn.functional.relu(x)
            x = self.dropout(x)
            layer_outputs.append(x)
        
        # 使用层间连接：将所有层的输出相加
        x = torch.stack(layer_outputs).sum(0)
        
        # 全局池化 - 确保输出384维特征
        x = torch_scatter.scatter_mean(x, batch, dim=0)
        x = self.global_pool_linear(x)
        
        # 输出层
        return self.out_layer(x)

# GIN卷积层实现
class GINLayer(torch.nn.Module):
    """
    GIN卷积层实现，包含：
    - 节点和边特征的拼接和MLP处理
    - MLP（多层感知器）
    - 残差连接
    - 可学习的epsilon参数
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 节点和边特征拼接后的MLP处理
        self.edge_node_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim * 2, in_dim),  # 拼接后维度翻倍，再降回原维度
            torch.nn.LayerNorm(in_dim),
            torch.nn.ReLU()
        )
        
        # MLP结构
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim),  # 第一层保持维度不变
            torch.nn.LayerNorm(in_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_dim, out_dim)  # 最后一层可以改变维度
        )
        
        # 如果输入输出维度不同，添加投影层
        self.projection = None
        if in_dim != out_dim:
            self.projection = torch.nn.Linear(in_dim, out_dim)
        
        self.eps = torch.nn.Parameter(torch.Tensor([0]))  # 可学习的epsilon参数

    def forward(self, x, edge_index, edge_attr):
        """前向传播"""
        # 保存输入用于残差连接
        identity = x
        
        row, col = edge_index  # 获取边的起点和终点
        x_j = x[row]  # 获取邻居节点特征
        
        # 将边特征与节点特征拼接而非相加，并通过MLP处理
        # 拼接操作 [x_j, edge_attr] 沿着特征维度
        x_combined = torch.cat([x_j, edge_attr], dim=1)
        # 通过MLP处理拼接后的特征
        x_j = self.edge_node_mlp(x_combined)
        
        # 聚合邻居信息并更新节点特征
        out = (1 + self.eps) * x + torch_scatter.scatter_add(x_j, col, dim=0, dim_size=x.size(0))
        out = self.mlp(out)
        
        # 应用残差连接
        if self.projection is not None:
            identity = self.projection(identity)
        out = out + identity
        
        return out

# 使用示例
if __name__ == "__main__":
    # 创建分子处理器
    processor = MolecularGraphProcessor()
    # 创建GIN模型
    model = GIN(
        atom_feature_dim=processor.atom_feature_dim,
        bond_feature_dim=processor.bond_feature_dim,
        hidden_dim=128,
        out_dim=256
    )

    # 多个SMILES字符串列表
    smiles_list = [
        "COC1=C(C=C2C(=C1)N=CN2C3=CC(=C(S3)C#N)OCC4=CC=CC=C4S(=O)(=O)C)OC",
        "CC(C)(C)c1ccc(C(=O)Nc2ccc(C(=O)O)cc2)cc1",
        "CCN(CC)CCNC(=O)c1ccc(N)cc1"
    ]
    
    # 处理多个分子
    mol_graphs = [processor.process_molecule(smiles) for smiles in smiles_list]
    
    # 创建批次数据
    batch_x = []
    batch_edge_index = []
    batch_edge_attr = []
    batch_ptr = [0]
    
    # 合并数据并创建batch索引
    node_offset = 0
    for graph in mol_graphs:
        batch_x.append(graph.x)
        
        # 调整边索引以考虑节点偏移
        edge_index = graph.edge_index.clone()
        edge_index[0] += node_offset
        edge_index[1] += node_offset
        batch_edge_index.append(edge_index)
        
        batch_edge_attr.append(graph.edge_attr)
        node_offset += graph.x.size(0)
        batch_ptr.append(node_offset)
    
    # 连接所有张量
    batch_x = torch.cat(batch_x, dim=0)
    batch_edge_index = torch.cat(batch_edge_index, dim=1)
    batch_edge_attr = torch.cat(batch_edge_attr, dim=0)
    
    # 创建batch索引张量
    batch = torch.zeros(batch_x.size(0), dtype=torch.long)
    for i in range(len(batch_ptr) - 1):
        batch[batch_ptr[i]:batch_ptr[i+1]] = i
    
    # 获取所有分子的全局特征
    mol_global_features = model(
        batch_x, 
        batch_edge_index,
        batch_edge_attr,
        batch
    )
    
    print(f"批次大小: {len(smiles_list)}")
    print(f"特征形状: {mol_global_features.shape}")
    
    # 也可以单独处理每个分子
    print("\n单独处理每个分子:")
    for i, smiles in enumerate(smiles_list):
        mol_graph = mol_graphs[i]
        mol_feature = model(
            mol_graph.x, 
            mol_graph.edge_index,
            mol_graph.edge_attr
        )
        print(f"分子 {i+1} 特征形状: {mol_feature.shape}")
