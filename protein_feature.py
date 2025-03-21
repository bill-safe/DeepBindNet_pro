# 导入必要的库
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm

class ResidualBlock1D(nn.Module):
    """
    一维残差块，包含两个卷积层和残差连接
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 如果输入和输出通道数不同，使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        
        # Dropout层
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # 第一个卷积层
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二个卷积层
        out = self.bn2(self.conv2(out))
        # 残差连接
        out += self.shortcut(x)
        # ReLU激活
        out = F.relu(out)
        # Dropout
        out = self.dropout(out)
        return out

class ResNet1D(nn.Module):
    """
    一维残差网络，用于处理蛋白质序列特征
    减少层数（4→3）以降低计算成本，提高训练稳定性
    """
    def __init__(self, in_channels, hidden_dims=[64, 128, 256], out_dim=384):
        super().__init__()
        # 输入卷积层
        self.conv1 = nn.Conv1d(in_channels, hidden_dims[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        # 残差层（减少到3层）
        self.layer1 = self._make_layer(hidden_dims[0], hidden_dims[0], stride=1)
        self.layer2 = self._make_layer(hidden_dims[0], hidden_dims[1], stride=2)
        self.layer3 = self._make_layer(hidden_dims[1], hidden_dims[2], stride=2)
        
        # 输出层
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dims[2], out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
    
    def _make_layer(self, in_channels, out_channels, stride):
        """创建残差层"""
        return ResidualBlock1D(in_channels, out_channels, stride)
    
    def forward(self, x):
        # 输入形状: [batch_size, sequence_length, in_channels]
        # 转换为卷积输入形状: [batch_size, in_channels, sequence_length]
        x = x.permute(0, 2, 1)
        
        # 输入卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 残差层（减少到3层）
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局平均池化
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # 输出层
        x = F.relu(self.bn2(self.fc(x)))
        
        return x

class ProteinFeatureExtractor(nn.Module):
    """
    蛋白质特征提取器，包含ESM预训练模型和ResNet1D
    使用分步降维（1280→512→128→64）减少信息丢失
    """
    def __init__(self, esm_model_path=None, esm_output_dim=1280, hidden_dims=[96, 192, 384], out_dim=384):
        super().__init__()
        # ESM模型
        self.esm_model_path = esm_model_path
        self.esm_output_dim = esm_output_dim
        
        # 创建模型目录
        model_dir = os.path.join(os.getcwd(), "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # 加载ESM模型
        if esm_model_path and os.path.exists(esm_model_path):
            # 从本地加载模型
            print(f"从本地加载ESM模型: {esm_model_path}")
            self.esm_model, self.esm_alphabet = torch.hub.load_state_dict_from_url(
                f"file://{esm_model_path}", map_location="cpu"
            )
        else:
            # 从PyTorch Hub下载模型到model文件夹
            print("从PyTorch Hub下载ESM模型到model文件夹...")
            
            # 设置torch.hub的下载目录
            torch.hub.set_dir(model_dir)
            
            # 下载模型
            self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            
            # 保存模型到model文件夹
            model_path = os.path.join(model_dir, "esm2_t33_650M_UR50D.pt")
            if not os.path.exists(model_path):
                print(f"保存模型到: {model_path}")
                torch.save({
                    'model': self.esm_model.state_dict(),
                    'alphabet': self.esm_alphabet
                }, model_path)
        
        # 冻结ESM模型参数
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        self.esm_model = self.esm_model.eval()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        
        # 分步降维投影层（1280→512→128→64）
        self.projection = nn.Sequential(
            nn.Linear(self.esm_output_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),  # 加入 LayerNorm，增强稳定性
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),  # 加入 LayerNorm
            nn.Linear(256, hidden_dims[0])
        )
        
        # ResNet1D（减少到3层）
        self.resnet = ResNet1D(
            in_channels=hidden_dims[0],
            hidden_dims=hidden_dims,
            out_dim=out_dim
        )
    
    def extract_esm_features(self, sequences):
        """
        使用ESM模型提取蛋白质序列特征

        参数:
        - sequences: 蛋白质序列列表

        返回:
        - 特征张量，形状为 [batch_size, sequence_length, esm_output_dim]
        """
        # 获取当前设备
        device = next(self.parameters()).device
        
        # 准备批次
        batch_data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]

        # 转换为ESM输入格式
        batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(batch_data)
        
        # 确保batch_tokens在正确的设备上
        batch_tokens = batch_tokens.to(device)
        
        # 确保ESM模型在正确的设备上
        self.esm_model = self.esm_model.to(device)

        # 使用ESM模型提取特征
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
        
        # 移除特殊标记（保留序列部分）
        features = []
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            # 提取序列部分（去除开始和结束标记）
            seq_features = token_representations[i, 1:seq_len+1]
            features.append(seq_features)
        
        # 填充到相同长度
        max_len = max(len(seq) for seq in sequences)
        padded_features = []
        
        for feat in features:
            # 计算需要填充的长度
            pad_len = max_len - feat.size(0)
            
            if pad_len > 0:
                # 在序列末尾填充零
                padding = torch.zeros(pad_len, self.esm_output_dim, device=feat.device)
                padded_feat = torch.cat([feat, padding], dim=0)
            else:
                padded_feat = feat
            
            padded_features.append(padded_feat)
        
        # 堆叠为批次
        return torch.stack(padded_features)
    
    def forward(self, sequences=None, features=None):
        """
        前向传播
        
        参数:
        - sequences: 蛋白质序列列表（如果提供）
        - features: 预计算的ESM特征（如果提供）
        
        返回:
        - 蛋白质特征向量，形状为 [batch_size, out_dim]
        """
        # 如果提供了序列，使用ESM提取特征
        if sequences is not None:
            features = self.extract_esm_features(sequences)
        
        # 如果既没有提供序列也没有提供特征，抛出错误
        if features is None:
            raise ValueError("必须提供序列或预计算的特征")
        
        # 投影到ResNet输入维度
        features = self.projection(features)
        
        # 通过ResNet处理
        return self.resnet(features)

# 使用示例
if __name__ == "__main__":
    # 创建蛋白质特征提取器
    extractor = ProteinFeatureExtractor(
        esm_model_path=None,  # 设置为None以触发自动下载
        out_dim=256
    )
    
    # 测试序列
    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
    ]
    
    # 提取特征
    features = extractor(sequences)
    print(f"输出特征形状: {features.shape}")  # 应该是 [2, 256]
