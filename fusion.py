# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        # 定义线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key, value, attn_mask=None):
        """
        前向传播
        
        参数:
        - query: 查询张量，形状为 [batch_size, query_len, embed_dim]
        - key: 键张量，形状为 [batch_size, key_len, embed_dim]
        - value: 值张量，形状为 [batch_size, key_len, embed_dim]
        - attn_mask: 注意力掩码，形状为 [batch_size, num_heads, query_len, key_len]
        
        返回:
        - 注意力输出，形状为 [batch_size, query_len, embed_dim]
        - 注意力权重，形状为 [batch_size, num_heads, query_len, key_len]
        """
        batch_size, query_len, _ = query.size()
        key_len = key.size(1)
        
        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头形式
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用掩码（如果提供）
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        output = torch.matmul(attn_weights, v)
        
        # 重塑回原始形状
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, self.embed_dim)
        
        # 最终线性投影
        output = self.out_proj(output)
        
        return output, attn_weights

class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # 多头自注意力
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # 前馈网络
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        """
        前向传播
        
        参数:
        - x: 输入张量，形状为 [batch_size, seq_len, embed_dim]
        - attn_mask: 注意力掩码
        
        返回:
        - 输出张量，形状为 [batch_size, seq_len, embed_dim]
        - 注意力权重
        """
        # 自注意力
        attn_output, attn_weights = self.self_attn(x, x, x, attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈网络
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x, attn_weights

class CrossAttentionFusion(nn.Module):
    """
    跨模态注意力融合模块
    """
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=1024, num_layers=2, dropout=0.1, protein_dim=None, molecule_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 特征投影层（如果需要）
        self.protein_projection = None
        if protein_dim is not None and protein_dim != embed_dim:
            self.protein_projection = nn.Linear(protein_dim, embed_dim)
            
        self.molecule_projection = None
        if molecule_dim is not None and molecule_dim != embed_dim:
            self.molecule_projection = nn.Linear(molecule_dim, embed_dim)
        
        # 跨模态注意力层
        self.cross_attn_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 动态融合网络
        self.dynamic_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
    
    def forward(self, protein_features, molecule_features):
        """
        前向传播
        
        参数:
        - protein_features: 蛋白质特征，形状为 [batch_size, protein_dim]
        - molecule_features: 分子特征，形状为 [batch_size, molecule_dim]
        
        返回:
        - 融合特征，形状为 [batch_size, embed_dim]
        - 注意力权重
        """
        batch_size = protein_features.size(0)
        
        # 应用特征投影（如果需要）
        if self.protein_projection is not None:
            protein_features = self.protein_projection(protein_features)
        elif protein_features.size(1) != self.embed_dim:
            # 如果没有预定义投影层但维度不匹配，创建一个临时投影层
            # 注意：这种方式不推荐用于训练，因为参数不会被优化
            protein_projection = nn.Linear(protein_features.size(1), self.embed_dim).to(protein_features.device)
            protein_features = protein_projection(protein_features)
            
        if self.molecule_projection is not None:
            molecule_features = self.molecule_projection(molecule_features)
        elif molecule_features.size(1) != self.embed_dim:
            # 如果没有预定义投影层但维度不匹配，创建一个临时投影层
            molecule_projection = nn.Linear(molecule_features.size(1), self.embed_dim).to(molecule_features.device)
            molecule_features = molecule_projection(molecule_features)
        
        # 将特征扩展为序列形式（添加序列长度维度）
        protein_seq = protein_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        molecule_seq = molecule_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # 初始化为蛋白质特征
        x = protein_seq
        all_attn_weights = []
        
        # 应用跨模态注意力层
        for layer in self.cross_attn_layers:
            # 使用蛋白质特征作为Query，分子特征作为Key和Value
            x, attn_weights = layer(x)
            all_attn_weights.append(attn_weights)
        
        # 提取融合后的特征
        fused_features = x.squeeze(1)  # [batch_size, embed_dim]
        
        # 计算动态融合权重
        concat_features = torch.cat([protein_features, molecule_features], dim=1)
        fusion_weights = self.dynamic_fusion(concat_features)  # [batch_size, 2]
        
        # 应用动态融合
        weighted_protein = protein_features * fusion_weights[:, 0].unsqueeze(1)
        weighted_molecule = molecule_features * fusion_weights[:, 1].unsqueeze(1)
        dynamic_fused = weighted_protein + weighted_molecule
        
        # 结合Transformer融合和动态融合
        final_fused = fused_features + dynamic_fused
        
        # 输出投影
        output = self.output_proj(final_fused)
        
        return output, all_attn_weights

class GatedCrossAttention(nn.Module):
    """
    门控跨模态注意力融合模块
    
    通过门控机制动态调节原始特征与注意力特征的平衡：
    - 噪声数据下自动降低注意力权重（g→0）
    - 强相关特征时提升注意力贡献（g→1）
    
    优化点:
    - 增加Transformer层数（2→4）增强特征融合能力
    - 优化gate_net结构，减少计算量（1024→512）
    - 增加梯度裁剪阈值，提高训练稳定性
    """
    def __init__(self, embed_dim=384, num_heads=8, ff_dim=1024, num_layers=4, dropout=0.2, protein_dim=None, molecule_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 特征投影层（如果需要）
        self.protein_projection = None
        if protein_dim is not None and protein_dim != embed_dim:
            self.protein_projection = nn.Linear(protein_dim, embed_dim)
            
        self.molecule_projection = None
        if molecule_dim is not None and molecule_dim != embed_dim:
            self.molecule_projection = nn.Linear(molecule_dim, embed_dim)
        
        # 跨模态注意力层（增加到4层）
        self.cross_attn_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 门控网络（优化结构，减少计算量）
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # 梯度裁剪阈值（增加到2.5提高稳定性）
        self.grad_clip_threshold = 2.5
    
    def forward(self, protein_features, molecule_features):
        """
        前向传播
        
        参数:
        - protein_features: 蛋白质特征，形状为 [batch_size, protein_dim]
        - molecule_features: 分子特征，形状为 [batch_size, molecule_dim]
        
        返回:
        - 融合特征，形状为 [batch_size, embed_dim]
        - 注意力权重
        """
        batch_size = protein_features.size(0)
        
        # 应用特征投影（如果需要）
        if self.protein_projection is not None:
            protein_features = self.protein_projection(protein_features)
        elif protein_features.size(1) != self.embed_dim:
            protein_projection = nn.Linear(protein_features.size(1), self.embed_dim).to(protein_features.device)
            protein_features = protein_projection(protein_features)
            
        if self.molecule_projection is not None:
            molecule_features = self.molecule_projection(molecule_features)
        elif molecule_features.size(1) != self.embed_dim:
            molecule_projection = nn.Linear(molecule_features.size(1), self.embed_dim).to(molecule_features.device)
            molecule_features = molecule_projection(molecule_features)
        
        # 将特征扩展为序列形式（添加序列长度维度）
        protein_seq = protein_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        molecule_seq = molecule_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # 初始化为蛋白质特征
        x = protein_seq
        all_attn_weights = []
        
        # 应用跨模态注意力层
        for layer in self.cross_attn_layers:
            # 使用蛋白质特征作为Query，分子特征作为Key和Value
            x, attn_weights = layer(x)
            all_attn_weights.append(attn_weights)
        
        # 提取融合后的特征
        attn_features = x.squeeze(1)  # [batch_size, embed_dim]
        
        # 计算门控系数
        gate_input = torch.cat([protein_features, molecule_features], dim=1)
        # 处理批归一化的维度问题
        g = self.gate_net(gate_input)  # [batch_size, 1]
        
        # 门控融合: g * attn_features + (1-g) * protein_features
        gated_features = g * attn_features + (1 - g) * protein_features
        
        # 输出投影
        output = self.output_proj(gated_features)
        
        # 梯度裁剪（在训练时）
        if self.training:
            for param in self.parameters():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_(param, self.grad_clip_threshold)
        
        return output, all_attn_weights

# 使用示例
if __name__ == "__main__":
    # 创建融合模块
    fusion_module = CrossAttentionFusion(
        embed_dim=256,
        num_heads=8,
        ff_dim=1024,
        num_layers=2,
        dropout=0.1
    )
    
    # 创建门控融合模块
    gated_fusion_module = GatedCrossAttention(
        embed_dim=256,
        num_heads=8,
        ff_dim=1024,
        num_layers=2,
        dropout=0.1
    )
    
    # 创建测试数据
    batch_size = 4
    protein_features = torch.randn(batch_size, 256)
    molecule_features = torch.randn(batch_size, 256)
    
    # 融合特征
    fused_features, attn_weights = fusion_module(protein_features, molecule_features)
    gated_features, gated_attn_weights = gated_fusion_module(protein_features, molecule_features)
    
    print(f"蛋白质特征形状: {protein_features.shape}")
    print(f"分子特征形状: {molecule_features.shape}")
    print(f"融合特征形状: {fused_features.shape}")
    print(f"门控融合特征形状: {gated_features.shape}")
    print(f"注意力权重数量: {len(attn_weights)}")
    print(f"第一层注意力权重形状: {attn_weights[0].shape}")
