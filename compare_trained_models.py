# 导入必要的库
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

from dataset import get_data_loaders
from model import DeepBindNet
from model_gated import DeepBindNetGated

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 检查是否有可用的中文字体
try:
    font_names = [f.name for f in mpl.font_manager.fontManager.ttflist]
    chinese_fonts = [f for f in font_names if 'SimHei' in f or 'Microsoft YaHei' in f or 'SimSun' in f or 'FangSong' in f]
    
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = [chinese_fonts[0]] + plt.rcParams['font.sans-serif']
    else:
        print("警告: 未找到中文字体，将使用英文标题")
        USE_ENGLISH = True
except:
    USE_ENGLISH = True
else:
    USE_ENGLISH = False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='比较标准模型和门控模型的性能')
    
    # 模型路径
    parser.add_argument('--standard_model_dir', type=str, default='outputs',
                        help='标准模型输出目录')
    parser.add_argument('--gated_model_dir', type=str, default='outputs_gated',
                        help='门控模型输出目录')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='预处理数据目录')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='评估设备')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='比较结果输出目录')
    
    return parser.parse_args()

def load_model(model_dir, device, is_gated=False):
    """加载模型"""
    # 加载检查点
    checkpoint_path = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到模型检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载scaler参数
    scaler_params = checkpoint.get('scaler_params')
    
    # 创建模型
    if is_gated:
        model = DeepBindNetGated(
            atom_feature_dim=6,
            bond_feature_dim=3,
            hidden_dim=256,  # 使用默认值，可以从checkpoint中恢复
            feature_dim=512,
            fusion_heads=8,
            fusion_layers=4,
            dropout_rate=0.1,
            scaler_params=scaler_params
        ).to(device)
    else:
        model = DeepBindNet(
            atom_feature_dim=6,
            bond_feature_dim=3,
            hidden_dim=256,
            feature_dim=512,
            fusion_heads=8,
            fusion_layers=4,
            dropout_rate=0.1,
            scaler_params=scaler_params
        ).to(device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def evaluate_model(model, test_loader, device, model_name="模型"):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 获取数据
            mol_graphs = batch['mol_graphs'].to(device)
            protein_features = batch['protein_features'].to(device)
            targets = batch['kiba_scores'].to(device)
            
            # 前向传播
            predictions, _, _ = model(mol_graphs, protein_features)
            
            # 如果有scaler参数，将targets反标准化以计算指标
            if model.scaler_params is not None:
                scale = torch.tensor(model.scaler_params['scale'], device=device)
                mean = torch.tensor(model.scaler_params['mean'], device=device)
                targets = targets * scale + mean
            
            # 转移到CPU并转换为numpy数组
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    # 打印评估摘要
    print(f'{model_name} 评估结果:')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  MAE: {mae:.4f}')
    print(f'  R²: {r2:.4f}')
    
    return {
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets),
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def plot_comparison(standard_results, gated_results, output_dir):
    """绘制比较图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 预测值与真实值散点图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(standard_results['targets'], standard_results['predictions'], alpha=0.5, s=10)
    plt.plot([min(standard_results['targets']), max(standard_results['targets'])],
             [min(standard_results['targets']), max(standard_results['targets'])],
             'r--', linewidth=2)
    plt.title('Standard Model Predictions' if USE_ENGLISH else '标准模型预测')
    plt.xlabel('True Values' if USE_ENGLISH else '真实值')
    plt.ylabel('Predicted Values' if USE_ENGLISH else '预测值')
    plt.text(0.05, 0.95, f'RMSE: {standard_results["rmse"]:.4f}\nR²: {standard_results["r2"]:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.subplot(1, 2, 2)
    plt.scatter(gated_results['targets'], gated_results['predictions'], alpha=0.5, s=10)
    plt.plot([min(gated_results['targets']), max(gated_results['targets'])],
             [min(gated_results['targets']), max(gated_results['targets'])],
             'r--', linewidth=2)
    plt.title('Gated Model Predictions' if USE_ENGLISH else '门控模型预测')
    plt.xlabel('True Values' if USE_ENGLISH else '真实值')
    plt.ylabel('Predicted Values' if USE_ENGLISH else '预测值')
    plt.text(0.05, 0.95, f'RMSE: {gated_results["rmse"]:.4f}\nR²: {gated_results["r2"]:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_comparison.png'), dpi=300)
    plt.close()
    
    # 2. 误差分布直方图
    plt.figure(figsize=(12, 5))
    
    standard_errors = standard_results['predictions'] - standard_results['targets']
    gated_errors = gated_results['predictions'] - gated_results['targets']
    
    plt.subplot(1, 2, 1)
    plt.hist(standard_errors, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.title('Standard Model Error Distribution' if USE_ENGLISH else '标准模型误差分布')
    plt.xlabel('Error' if USE_ENGLISH else '误差')
    plt.ylabel('Frequency' if USE_ENGLISH else '频率')
    
    plt.subplot(1, 2, 2)
    plt.hist(gated_errors, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.title('Gated Model Error Distribution' if USE_ENGLISH else '门控模型误差分布')
    plt.xlabel('Error' if USE_ENGLISH else '误差')
    plt.ylabel('Frequency' if USE_ENGLISH else '频率')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300)
    plt.close()
    
    # 3. 性能指标比较条形图
    plt.figure(figsize=(10, 6))
    
    metrics = ['RMSE', 'MAE', '1 - R²']
    standard_values = [standard_results['rmse'], standard_results['mae'], 1 - standard_results['r2']]
    gated_values = [gated_results['rmse'], gated_results['mae'], 1 - gated_results['r2']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, standard_values, width, label='Standard Model' if USE_ENGLISH else '标准模型')
    plt.bar(x + width/2, gated_values, width, label='Gated Model' if USE_ENGLISH else '门控模型')
    
    plt.xlabel('Metrics' if USE_ENGLISH else '评估指标')
    plt.ylabel('Value' if USE_ENGLISH else '值')
    plt.title('Performance Comparison' if USE_ENGLISH else '性能比较')
    plt.xticks(x, metrics)
    plt.legend()
    
    # 添加数值标签
    for i, v in enumerate(standard_values):
        plt.text(i - width/2, v + 0.02, f'{v:.4f}', ha='center')
    
    for i, v in enumerate(gated_values):
        plt.text(i + width/2, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
    plt.close()
    
    # 4. 性能改进百分比
    improvement_rmse = (standard_results['rmse'] - gated_results['rmse']) / standard_results['rmse'] * 100
    improvement_mae = (standard_results['mae'] - gated_results['mae']) / standard_results['mae'] * 100
    improvement_r2 = (gated_results['r2'] - standard_results['r2']) / (1 - standard_results['r2']) * 100 if standard_results['r2'] < 1 else 0
    
    plt.figure(figsize=(8, 6))
    
    metrics = ['RMSE', 'MAE', 'R²']
    improvements = [improvement_rmse, improvement_mae, improvement_r2]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    plt.bar(metrics, improvements, color=colors)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.xlabel('Metrics' if USE_ENGLISH else '评估指标')
    plt.ylabel('Improvement (%)' if USE_ENGLISH else '改进百分比 (%)')
    plt.title('Performance Improvement of Gated Model' if USE_ENGLISH else '门控模型性能改进')
    
    # 添加数值标签
    for i, v in enumerate(improvements):
        plt.text(i, v + np.sign(v) * 1, f'{v:.2f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_percentage.png'), dpi=300)
    plt.close()
    
    print(f"比较图表已保存到目录: {output_dir}")
    print(f"门控模型相对于标准模型的改进:")
    print(f"  RMSE: {improvement_rmse:.2f}%")
    print(f"  MAE: {improvement_mae:.2f}%")
    print(f"  R²: {improvement_r2:.2f}%")

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f'使用设备: {device}')
    
    # 加载数据
    _, _, test_loader = get_data_loaders(
        os.path.join(args.data_dir, 'train_data.pkl'),
        os.path.join(args.data_dir, 'val_data.pkl'),
        os.path.join(args.data_dir, 'test_data.pkl'),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 加载标准模型
    print("\n加载标准模型...")
    try:
        standard_model, standard_checkpoint = load_model(args.standard_model_dir, device, is_gated=False)
        print(f"成功加载标准模型，最佳验证RMSE: {standard_checkpoint.get('val_rmse', 'N/A')}")
    except Exception as e:
        print(f"加载标准模型失败: {e}")
        return
    
    # 加载门控模型
    print("\n加载门控模型...")
    try:
        gated_model, gated_checkpoint = load_model(args.gated_model_dir, device, is_gated=True)
        print(f"成功加载门控模型，最佳验证RMSE: {gated_checkpoint.get('val_rmse', 'N/A')}")
    except Exception as e:
        print(f"加载门控模型失败: {e}")
        return
    
    # 评估标准模型
    print("\n评估标准模型...")
    standard_results = evaluate_model(standard_model, test_loader, device, "标准模型")
    
    # 评估门控模型
    print("\n评估门控模型...")
    gated_results = evaluate_model(gated_model, test_loader, device, "门控模型")
    
    # 绘制比较图表
    print("\n生成比较图表...")
    plot_comparison(standard_results, gated_results, args.output_dir)
    
    # 保存评估结果
    results = {
        'standard': standard_results,
        'gated': gated_results
    }
    with open(os.path.join(args.output_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("\n比较完成!")

if __name__ == '__main__':
    main()
