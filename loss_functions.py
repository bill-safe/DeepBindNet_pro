# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogCoshLoss(nn.Module):
    """
    Log-Cosh 损失函数
    
    Log-Cosh 是一个平滑的损失函数，对大误差不敏感，可以减轻极端值的影响。
    公式: log(cosh(predictions - targets))
    
    特点:
    - 对小误差近似为MSE
    - 对大误差近似为MAE，但更平滑
    - 处处二阶可导
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        """
        计算 Log-Cosh 损失
        
        参数:
        - predictions: 模型预测值
        - targets: 目标值
        
        返回:
        - 损失值
        """
        diff = predictions - targets
        loss = torch.log(torch.cosh(diff + 1e-12))
        return torch.mean(loss)


class WeightedLoss(nn.Module):
    """
    基于目标值的加权损失函数
    
    对训练集中落在目标值分布两端的样本加权，避免模型偏向分布主区间。
    
    特点:
    - 可以与任何基础损失函数结合使用
    - 支持多种权重计算方式
    - 可以调整权重强度
    """
    def __init__(self, base_criterion, weight_type='abs_diff', alpha=1.0, normalize=True):
        """
        初始化加权损失函数
        
        参数:
        - base_criterion: 基础损失函数，如nn.MSELoss()、nn.L1Loss()等
        - weight_type: 权重计算方式，可选'abs_diff'、'squared_diff'、'exp_diff'
        - alpha: 权重强度系数，值越大权重差异越明显
        - normalize: 是否对权重进行归一化
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.weight_type = weight_type
        self.alpha = alpha
        self.normalize = normalize
        
    def forward(self, predictions, targets):
        """
        计算加权损失
        
        参数:
        - predictions: 模型预测值
        - targets: 目标值
        
        返回:
        - 加权损失值
        """
        # 计算基础损失
        base_loss = self.base_criterion(predictions, targets)
        
        if isinstance(base_loss, torch.Tensor) and base_loss.dim() > 0:
            # 如果基础损失函数返回每个样本的损失
            element_wise_loss = base_loss
        else:
            # 如果基础损失函数返回标量，则需要重新计算每个样本的损失
            if isinstance(self.base_criterion, nn.MSELoss):
                element_wise_loss = F.mse_loss(predictions, targets, reduction='none')
            elif isinstance(self.base_criterion, nn.L1Loss):
                element_wise_loss = F.l1_loss(predictions, targets, reduction='none')
            elif isinstance(self.base_criterion, nn.HuberLoss):
                element_wise_loss = F.huber_loss(predictions, targets, reduction='none', 
                                                delta=self.base_criterion.delta)
            elif isinstance(self.base_criterion, LogCoshLoss):
                diff = predictions - targets
                element_wise_loss = torch.log(torch.cosh(diff + 1e-12))
            else:
                # 对于其他损失函数，默认使用MSE
                element_wise_loss = F.mse_loss(predictions, targets, reduction='none')
        
        # 计算目标值与均值的差异
        targets_mean = torch.mean(targets)
        diff_from_mean = targets - targets_mean
        
        # 根据权重类型计算权重
        if self.weight_type == 'abs_diff':
            # 使用绝对差异作为权重
            weights = torch.abs(diff_from_mean) ** self.alpha
        elif self.weight_type == 'squared_diff':
            # 使用平方差异作为权重
            weights = diff_from_mean ** 2 ** (self.alpha / 2)
        elif self.weight_type == 'exp_diff':
            # 使用指数差异作为权重
            weights = torch.exp(self.alpha * torch.abs(diff_from_mean) / torch.max(torch.abs(diff_from_mean)))
        else:
            raise ValueError(f"不支持的权重类型: {self.weight_type}")
        
        # 归一化权重
        if self.normalize:
            weights = weights / (torch.sum(weights) + 1e-12) * len(weights)
        
        # 计算加权损失
        weighted_loss = torch.mean(weights * element_wise_loss)
        
        return weighted_loss


class GaussianNLLLoss(nn.Module):
    """
    高斯负对数似然损失函数
    
    用于联合建模预测均值和方差，使模型能够输出预测区间。
    公式: log(sigma) + 0.5 * ((target - mu)^2 / sigma^2)
    
    特点:
    - 允许模型预测不确定性
    - 对不同样本自适应调整损失权重
    - 防止模型过度自信
    """
    def __init__(self, eps=1e-6, reduction='mean'):
        """
        初始化高斯NLL损失函数
        
        参数:
        - eps: 数值稳定性常数
        - reduction: 损失聚合方式，可选'none'、'mean'、'sum'
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        
    def forward(self, mu, log_sigma, targets):
        """
        计算高斯NLL损失
        
        参数:
        - mu: 预测均值
        - log_sigma: 预测对数标准差
        - targets: 目标值
        
        返回:
        - 损失值
        """
        # 计算标准差
        sigma = torch.exp(log_sigma) + self.eps
        
        # 计算负对数似然
        loss = log_sigma + 0.5 * ((targets - mu) ** 2) / (sigma ** 2)
        
        # 应用reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError(f"不支持的reduction类型: {self.reduction}")
