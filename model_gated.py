# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
from gin import GIN
from protein_feature import ProteinFeatureExtractor
from fusion import GatedCrossAttention

class DeepBindNetGated(nn.Module):
    """
    DeepBindNet模型的门控版本，用于预测蛋白质-小分子结合亲和力
    
    包含:
    - 分子特征提取模块 (GIN)：5层残差GIN卷积，输出384维特征
    - 蛋白质特征提取模块 (ESM + ResNet1D)：3层ResNet1D，输出384维特征
    - 门控跨模态融合模块 (Gated Cross-Attention)：4层Transformer，优化的门控网络
    - 预测模块 (全连接层)：增加Dropout到0.2，减少过拟合
    """
    def __init__(self, 
                 atom_feature_dim=6, 
                 bond_feature_dim=3,
                 hidden_dim=128,
                 feature_dim=384,  # 更新为384维特征
                 esm_model_path=None,
                 esm_output_dim=1280,
                 fusion_heads=8,
                 fusion_layers=4,  # 增加到4层
                 dropout_rate=0.2,
                 scaler_params=None):
        super().__init__()
        self.scaler_params = scaler_params
        
        # 分子特征提取模块 (GIN)
        self.molecule_encoder = GIN(
            atom_feature_dim=atom_feature_dim,
            bond_feature_dim=bond_feature_dim,
            hidden_dim=hidden_dim,
            out_dim=feature_dim,
            dropout_rate=dropout_rate
        )
        
        # 蛋白质特征提取模块 (ESM + ResNet1D)
        self.protein_encoder = ProteinFeatureExtractor(
            esm_model_path=esm_model_path,
            esm_output_dim=esm_output_dim,
            hidden_dims=[128, 192, 256],  # 减少到3层
            out_dim=feature_dim
        )
        
        # 门控跨模态融合模块 (Gated Cross-Attention)
        self.fusion_module = GatedCrossAttention(
            embed_dim=feature_dim,
            num_heads=fusion_heads,
            ff_dim=feature_dim * 4,
            num_layers=fusion_layers,
            dropout=dropout_rate
        )
        
        # 预测模块 (全连接层)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 增加Dropout到0.2
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, mol_graphs, protein_features=None, protein_sequences=None):
        """
        前向传播
        
        参数:
        - mol_graphs: 分子图数据，包含x, edge_index, edge_attr和batch
        - protein_features: 预计算的蛋白质特征 (如果提供)
        - protein_sequences: 蛋白质序列列表 (如果提供)
        
        返回:
        - 预测的KIBA分数
        - 融合特征
        - 注意力权重
        """
        # 提取分子特征
        molecule_features = self.molecule_encoder(
            mol_graphs.x, 
            mol_graphs.edge_index,
            mol_graphs.edge_attr,
            mol_graphs.batch
        )
        
        # 提取蛋白质特征
        if protein_features is None and protein_sequences is None:
            raise ValueError("必须提供protein_features或protein_sequences")
        
        if protein_features is None:
            protein_features = self.protein_encoder(sequences=protein_sequences)
        
        # 融合特征 (使用门控跨模态注意力)
        fused_features, attn_weights = self.fusion_module(protein_features, molecule_features)
        
        # 预测KIBA分数
        predictions = self.predictor(fused_features).squeeze(-1)
        
        # 如果在评估/预测模式下且有scaler参数，进行反标准化
        if not self.training and self.scaler_params is not None:
            scale = torch.tensor(self.scaler_params['scale'], device=predictions.device)
            mean = torch.tensor(self.scaler_params['mean'], device=predictions.device)
            predictions = predictions * scale + mean
        
        return predictions, fused_features, attn_weights
    
    def predict(self, mol_graphs, protein_features=None, protein_sequences=None):
        """
        预测函数，仅返回预测的KIBA分数
        """
        predictions, _, _ = self.forward(mol_graphs, protein_features, protein_sequences)
        return predictions

# 使用示例
if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    
    # 创建DeepBindNetGated模型
    model = DeepBindNetGated(
        atom_feature_dim=6,
        bond_feature_dim=3,
        hidden_dim=128,
        feature_dim=256,
        esm_model_path=None,  # 设置为None以触发自动下载
        fusion_heads=8,
        fusion_layers=2,
        dropout_rate=0.2
    )
    
    # 创建测试数据
    batch_size = 2
    
    # 创建分子图
    x1 = torch.randn(5, 6)  # 5个节点，每个节点6维特征
    edge_index1 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                               [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    edge_attr1 = torch.randn(8, 3)  # 8条边，每条边3维特征
    
    x2 = torch.randn(4, 6)  # 4个节点，每个节点6维特征
    edge_index2 = torch.tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    edge_attr2 = torch.randn(6, 3)  # 6条边，每条边3维特征
    
    data_list = [
        Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1),
        Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)
    ]
    
    mol_graphs = Batch.from_data_list(data_list)
    
    # 创建蛋白质特征
    protein_features = torch.randn(batch_size, 256)
    
    # 前向传播
    predictions, fused_features, attn_weights = model(mol_graphs, protein_features)
    
    print(f"分子图节点数: {mol_graphs.x.shape[0]}")
    print(f"分子图边数: {mol_graphs.edge_index.shape[1]}")
    print(f"蛋白质特征形状: {protein_features.shape}")
    print(f"预测KIBA分数形状: {predictions.shape}")
    print(f"融合特征形状: {fused_features.shape}")
    print(f"注意力权重数量: {len(attn_weights)}")
    print(f"使用门控跨模态注意力融合")
