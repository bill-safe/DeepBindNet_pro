# 导入必要的库
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, r2_score
import pickle

from dataset import get_data_loaders
from model_gated import DeepBindNetGated  # 导入门控版本的模型
from lookahead import Lookahead  # 导入Lookahead优化器

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepBindNet门控版本训练脚本')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='预处理数据目录')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='隐藏层维度')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='特征维度')
    parser.add_argument('--fusion_heads', type=int, default=8,
                        help='融合模块注意力头数')
    parser.add_argument('--fusion_layers', type=int, default=5,
                        help='融合模块层数')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout比率')
    parser.add_argument('--esm_model_path', type=str, default=None,
                        help='ESM预训练模型路径，如果为None则自动下载')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=0.008,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--lookahead_k', type=int, default=5,
                        help='Lookahead优化器k步参数')
    parser.add_argument('--lookahead_alpha', type=float, default=0.5,
                        help='Lookahead优化器alpha参数')
    parser.add_argument('--t_max', type=int, default=50,
                        help='CosineAnnealingLR的T_max参数')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    parser.add_argument('--output_dir', type=str, default='outputs_gated',  # 默认输出到不同目录
                        help='输出目录')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志打印间隔（批次）')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='模型保存间隔（轮）')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='是否使用混合精度训练')
    
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, args, scaler=None):
    """训练一个轮次"""
    model.train()
    epoch_loss = 0
    all_preds = []
    all_targets = []
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # 获取数据
        mol_graphs = batch['mol_graphs'].to(device)
        protein_features = batch['protein_features'].to(device)
        targets = batch['kiba_scores'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播（使用混合精度训练）
        if args.mixed_precision and scaler is not None:
            with autocast():
                predictions, _, _ = model(mol_graphs, protein_features)
                loss = criterion(predictions, targets)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 前向传播
            predictions, _, _ = model(mol_graphs, protein_features)
            loss = criterion(predictions, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
        
        # 记录损失和预测
        epoch_loss += loss.item()
        all_preds.extend(predictions.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
        
        # 打印日志
        if (batch_idx + 1) % args.log_interval == 0:
            print(f'Epoch: {epoch+1}/{args.num_epochs} [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f}')
    
    # 计算指标
    epoch_loss /= len(train_loader)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('RMSE/train', rmse, epoch)
    writer.add_scalar('R2/train', r2, epoch)
    
    # 打印轮次摘要
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, '
          f'Time: {epoch_time:.2f}s')
    
    return epoch_loss, rmse, r2

def validate(model, val_loader, criterion, device, epoch, writer):
    """验证模型"""
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            # 获取数据
            mol_graphs = batch['mol_graphs'].to(device)
            protein_features = batch['protein_features'].to(device)
            targets = batch['kiba_scores'].to(device)
            
            # 前向传播（模型会自动处理反标准化）
            predictions, _, _ = model(mol_graphs, protein_features)
            
            # 计算损失（使用标准化的值）
            if not model.training and model.scaler_params is not None:
                # 将预测值重新标准化以计算损失
                scale = torch.tensor(model.scaler_params['scale'], device=device)
                mean = torch.tensor(model.scaler_params['mean'], device=device)
                predictions_std = (predictions - mean) / scale
                loss = criterion(predictions_std, targets)
            else:
                loss = criterion(predictions, targets)
            
            # 记录损失和预测
            val_loss += loss.item()
            
            # 如果有scaler参数，将targets反标准化以计算指标
            if model.scaler_params is not None:
                targets = targets * scale + mean
            
            # 转移到CPU并转换为numpy数组
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算指标
    val_loss /= len(val_loader)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('RMSE/val', rmse, epoch)
    writer.add_scalar('R2/val', r2, epoch)
    
    # 打印验证摘要
    print(f'Validation - Loss: {val_loss:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}')
    
    return val_loss, rmse, r2

def test(model, test_loader, criterion, device):
    """测试模型"""
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    all_smiles = []
    all_proteins = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 获取数据
            mol_graphs = batch['mol_graphs'].to(device)
            protein_features = batch['protein_features'].to(device)
            targets = batch['kiba_scores'].to(device)
            
            # 前向传播（模型会自动处理反标准化）
            predictions, _, _ = model(mol_graphs, protein_features)
            
            # 计算损失（使用标准化的值）
            if not model.training and model.scaler_params is not None:
                # 将预测值重新标准化以计算损失
                scale = torch.tensor(model.scaler_params['scale'], device=device)
                mean = torch.tensor(model.scaler_params['mean'], device=device)
                predictions_std = (predictions - mean) / scale
                loss = criterion(predictions_std, targets)
            else:
                loss = criterion(predictions, targets)
            
            # 记录损失和预测
            test_loss += loss.item()
            
            # 如果有scaler参数，将targets反标准化以计算指标
            if model.scaler_params is not None:
                targets = targets * scale + mean
            
            # 转移到CPU并转换为numpy数组
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_smiles.extend(batch['smiles'])
            all_proteins.extend(batch['protein_sequences'])
    
    # 计算指标
    test_loss /= len(test_loader)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    # 打印测试摘要
    print(f'Test - Loss: {test_loss:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}')
    
    # 返回测试结果
    return {
        'loss': test_loss,
        'rmse': rmse,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets,
        'smiles': all_smiles,
        'proteins': all_proteins
    }

def save_checkpoint(model, optimizer, scheduler, epoch, val_rmse, best_rmse, output_dir, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_rmse': val_rmse,
        'best_rmse': best_rmse,
        'scaler_params': model.scaler_params  # 保存scaler参数
    }
    
    # 保存最新检查点
    torch.save(checkpoint, os.path.join(output_dir, 'latest_checkpoint.pt'))
    
    # 如果是最佳模型，也保存为best_model.pt
    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, 'best_model.pt'))
        print(f'保存最佳模型，RMSE: {val_rmse:.4f}')
    
    # 每隔一定轮次保存一次
    if (epoch + 1) % 5 == 0:
        torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f'使用设备: {device}')
    print(f'使用门控跨模态注意力机制')
    
    # 设置TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # 加载数据
    train_path = os.path.join(args.data_dir, 'train_data.pkl')
    val_path = os.path.join(args.data_dir, 'val_data.pkl')
    test_path = os.path.join(args.data_dir, 'test_data.pkl')
    
    # 加载scaler参数
    scaler_path = os.path.join(args.data_dir, 'scaler_params.pkl')
    if os.path.exists(scaler_path):
        print(f"加载scaler参数: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler_params = pickle.load(f)
    else:
        print("警告: 未找到scaler参数文件，将不进行反标准化")
        scaler_params = None
    
    train_loader, val_loader, test_loader = get_data_loaders(
        train_path, val_path, test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 创建门控版本的模型
    model = DeepBindNetGated(
        atom_feature_dim=6,
        bond_feature_dim=3,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        esm_model_path=args.esm_model_path,
        fusion_heads=args.fusion_heads,
        fusion_layers=args.fusion_layers,
        dropout_rate=args.dropout_rate,
        scaler_params=scaler_params
    ).to(device)
    
    # 打印模型结构
    print(model)
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 定义优化器 (AdamW)
    base_optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 定义学习率调度器 (CosineAnnealingLR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer,
        T_max=args.t_max,
        eta_min=1e-6
    )
    
    # 包装为Lookahead优化器
    optimizer = Lookahead(
        base_optimizer,
        k=args.lookahead_k,
        alpha=args.lookahead_alpha
    )
    
    # 混合精度训练
    scaler = GradScaler() if args.mixed_precision else None
    
    # 训练循环
    best_val_rmse = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        # 训练一个轮次
        train_loss, train_rmse, train_r2 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, args, scaler
        )
        
        # 验证
        val_loss, val_rmse, val_r2 = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # 更新学习率
        scheduler.step()
        
        # 检查是否是最佳模型
        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 保存检查点
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_rmse, best_val_rmse, args.output_dir, is_best
        )
        
        # 早停
        if patience_counter >= args.patience:
            print(f'早停! {args.patience} 轮未改善')
            break
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试最佳模型
    test_results = test(model, test_loader, criterion, device)
    
    # 保存测试结果
    with open(os.path.join(args.output_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results, f)
    
    # 打印最终结果
    print(f"训练完成! 最佳验证RMSE: {best_val_rmse:.4f}, 测试RMSE: {test_results['rmse']:.4f}")
    
    # 关闭TensorBoard
    writer.close()

if __name__ == '__main__':
    main()
