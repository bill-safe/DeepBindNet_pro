# 导入必要的库
import os
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde

# 尝试导入seaborn，如果不可用则使用matplotlib默认样式
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("警告: seaborn未安装，将使用matplotlib默认样式")

# 设置中文字体
def set_chinese_font():
    """设置支持中文的字体"""
    # 常见的中文字体列表
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'NSimSun', 'FangSong', 'KaiTi', 'SimSun']
    
    # 检查系统中是否有这些字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 尝试找到一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = [font]
            print(f"使用中文字体: {font}")
            # 解决负号显示问题
            plt.rcParams['axes.unicode_minus'] = False
            return True
    
    # 如果没有找到中文字体，尝试使用系统默认字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("尝试使用 SimHei 字体")
        return True
    except:
        print("警告: 未找到支持中文的字体，图表中的中文可能无法正确显示")
        return False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepBindNet预测可视化脚本')
    
    parser.add_argument('--results_file', type=str, required=True,
                        help='测试结果文件路径 (pickle格式)')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                        help='可视化输出目录')
    parser.add_argument('--num_bins', type=int, default=5,
                        help='分析区间数量')
    
    return parser.parse_args()

def load_results(results_file):
    """加载测试结果"""
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    return results

def calculate_ci(y_true, y_pred):
    """计算一致性指数 (Concordance Index, CI)
    
    CI衡量模型预测值与真实值之间的相对排序一致性。
    CI = 1 表示预测顺序完全一致（完美排序）
    CI = 0.5 表示随机排序（没有排序能力）
    CI = 0 表示排序完全相反（最差情况）
    
    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        
    返回:
        ci: 一致性指数值
    """
    n = len(y_true)
    # 初始化计数器
    concordant = 0
    total_pairs = 0
    
    # 遍历所有可能的样本对
    for i in range(n):
        for j in range(i+1, n):
            if y_true[i] != y_true[j]:  # 只考虑真实值不相等的情况
                total_pairs += 1
                
                # 如果真实值的排序与预测值的排序一致
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant += 1
                # 如果预测值相等，算作0.5个一致对
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
    
    # 如果没有有效的样本对，返回0.5（随机猜测）
    if total_pairs == 0:
        return 0.5
    
    return concordant / total_pairs

def plot_predictions_vs_targets(predictions, targets, output_dir):
    """绘制预测值vs实际值散点图"""
    plt.figure(figsize=(10, 8))
    
    # 散点图
    plt.scatter(targets, predictions, alpha=0.5, s=20)
    
    # 添加对角线 (y=x)
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测线 (y=x)')
    
    # 添加回归线
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    plt.plot(targets, p(targets), 'b-', label=f'回归线 (y={z[0]:.2f}x+{z[1]:.2f})')
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    ci = calculate_ci(targets, predictions)
    
    # 添加标题和标签
    plt.title(f'预测值 vs 实际值\nRMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}, CI={ci:.4f}')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 特别处理R²图表的负号问题
    ax = plt.gca()
    # 使用ASCII负号代替Unicode负号
    for tick in ax.get_yticklabels():
        tick.set_text(tick.get_text().replace('−', '-'))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_targets.png'), dpi=300)
    plt.close()

def plot_residuals(predictions, targets, output_dir):
    """绘制残差图"""
    residuals = predictions - targets
    
    plt.figure(figsize=(10, 8))
    
    # 残差散点图
    plt.scatter(targets, residuals, alpha=0.5, s=20)
    
    # 添加水平线 (y=0)
    plt.axhline(y=0, color='r', linestyle='--', label='零残差线')
    
    # 添加回归线
    z = np.polyfit(targets, residuals, 1)
    p = np.poly1d(z)
    plt.plot(targets, p(targets), 'b-', label=f'趋势线 (y={z[0]:.2f}x+{z[1]:.2f})')
    
    # 计算残差统计量
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # 添加标题和标签
    plt.title(f'残差图\n均值={mean_residual:.4f}, 标准差={std_residual:.4f}')
    plt.xlabel('实际值')
    plt.ylabel('残差 (预测值 - 实际值)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 特别处理负号问题
    ax = plt.gca()
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_text(tick.get_text().replace('−', '-'))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals.png'), dpi=300)
    plt.close()
    
    # 绘制残差直方图
    plt.figure(figsize=(10, 6))
    
    # 残差直方图
    plt.hist(residuals, bins=30, alpha=0.7, density=True)
    
    # 添加正态分布曲线
    x = np.linspace(min(residuals), max(residuals), 100)
    plt.plot(x, 1/(std_residual * np.sqrt(2 * np.pi)) * 
             np.exp(-(x - mean_residual)**2 / (2 * std_residual**2)), 
             'r-', label='正态分布拟合')
    
    # 添加标题和标签
    plt.title(f'残差分布\n均值={mean_residual:.4f}, 标准差={std_residual:.4f}')
    plt.xlabel('残差 (预测值 - 实际值)')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 特别处理负号问题
    ax = plt.gca()
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_text(tick.get_text().replace('−', '-'))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_histogram.png'), dpi=300)
    plt.close()

def plot_distribution_comparison(predictions, targets, output_dir):
    """绘制预测值分布与实际值分布的比较"""
    plt.figure(figsize=(12, 6))
    
    # 计算KDE
    kde_targets = gaussian_kde(targets)
    kde_predictions = gaussian_kde(predictions)
    
    # 生成x轴值
    x = np.linspace(min(min(targets), min(predictions)), 
                    max(max(targets), max(predictions)), 1000)
    
    # 绘制KDE曲线
    plt.plot(x, kde_targets(x), 'b-', label='实际值分布', linewidth=2)
    plt.plot(x, kde_predictions(x), 'r-', label='预测值分布', linewidth=2)
    
    # 添加均值线
    plt.axvline(x=np.mean(targets), color='b', linestyle='--', 
                label=f'实际值均值: {np.mean(targets):.4f}')
    plt.axvline(x=np.mean(predictions), color='r', linestyle='--', 
                label=f'预测值均值: {np.mean(predictions):.4f}')
    
    # 计算分布统计量
    target_std = np.std(targets)
    pred_std = np.std(predictions)
    
    # 添加标题和标签
    plt.title(f'预测值与实际值分布比较\n实际值标准差={target_std:.4f}, 预测值标准差={pred_std:.4f}')
    plt.xlabel('值')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 特别处理负号问题
    ax = plt.gca()
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_text(tick.get_text().replace('−', '-'))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'), dpi=300)
    plt.close()

def analyze_by_range(predictions, targets, num_bins, output_dir):
    """按值域范围分析预测性能"""
    # 计算目标值的分位数
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(targets, quantiles)
    
    # 初始化结果
    bin_names = []
    bin_counts = []
    bin_rmses = []
    bin_maes = []
    bin_r2s = []
    bin_cis = []  # 一致性指数
    bin_biases = []  # 平均偏差 (预测值 - 实际值)
    
    # 对每个区间计算指标
    for i in range(num_bins):
        bin_name = f"Bin {i+1} [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]"
        
        # 获取当前区间的样本
        mask = (targets >= bin_edges[i]) & (targets <= bin_edges[i+1])
        bin_targets = targets[mask]
        bin_preds = predictions[mask]
        
        # 如果区间内有样本，计算指标
        if len(bin_targets) > 0:
            bin_rmse = np.sqrt(mean_squared_error(bin_targets, bin_preds))
            bin_mae = mean_absolute_error(bin_targets, bin_preds)
            bin_r2 = r2_score(bin_targets, bin_preds) if len(bin_targets) > 1 else float('nan')
            bin_ci = calculate_ci(bin_targets, bin_preds) if len(bin_targets) > 1 else float('nan')
            bin_bias = np.mean(bin_preds - bin_targets)
            bin_count = len(bin_targets)
            
            bin_names.append(bin_name)
            bin_counts.append(bin_count)
            bin_rmses.append(bin_rmse)
            bin_maes.append(bin_mae)
            bin_r2s.append(bin_r2)
            bin_cis.append(bin_ci)
            bin_biases.append(bin_bias)
    
    # 创建DataFrame
    results_df = pd.DataFrame({
        'Bin': bin_names,
        'Count': bin_counts,
        'Percentage': [count / len(targets) * 100 for count in bin_counts],
        'RMSE': bin_rmses,
        'MAE': bin_maes,
        'R2': bin_r2s,
        'CI': bin_cis,
        'Bias': bin_biases
    })
    
    # 保存结果
    results_df.to_csv(os.path.join(output_dir, 'range_analysis.csv'), index=False)
    
    # 绘制区间RMSE柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(bin_names, bin_rmses, alpha=0.7)
    plt.axhline(y=np.sqrt(mean_squared_error(targets, predictions)), 
                color='r', linestyle='--', label='整体RMSE')
    plt.title('各值域区间的RMSE')
    plt.xlabel('值域区间')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 特别处理负号问题
    ax = plt.gca()
    for tick in ax.get_yticklabels():
        tick.set_text(tick.get_text().replace('−', '-'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'range_rmse.png'), dpi=300)
    plt.close()
    
    # 绘制区间偏差柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(bin_names, bin_biases, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', label='零偏差线')
    plt.axhline(y=np.mean(predictions - targets), 
                color='r', linestyle='--', label='整体平均偏差')
    plt.title('各值域区间的平均偏差 (预测值 - 实际值)')
    plt.xlabel('值域区间')
    plt.ylabel('平均偏差')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 特别处理负号问题
    ax = plt.gca()
    for tick in ax.get_yticklabels():
        tick.set_text(tick.get_text().replace('−', '-'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'range_bias.png'), dpi=300)
    plt.close()
    
    return results_df

def analyze_by_affinity_level(predictions, targets, output_dir):
    """按亲和力等级分析预测性能
    
    根据 log(Ki) 预测结果将数据划分为五个亲和力等级区间，并统计每个等级的预测性能
    区间如下（单位：nM）：
        - 极强结合: Ki < 10 nM      -> log(Ki) < 1
        - 强结合:   10 ~ 100 nM     -> 1 <= log(Ki) < 2
        - 中等结合: 100 ~ 1000 nM   -> 2 <= log(Ki) < 3
        - 弱结合:   1 ~ 100 μM      -> 3 <= log(Ki) < 5
        - 极弱结合: Ki > 100 μM     -> log(Ki) >= 5
    """
    # 定义亲和力等级区间
    bin_edges = [float('-inf'), 1, 2, 3, 5, float('inf')]
    bin_labels = [
        "极强结合 (Ki < 10 nM)",
        "强结合 (10 ~ 100 nM)",
        "中等结合 (100 ~ 1000 nM)",
        "弱结合 (1 ~ 100 μM)",
        "极弱结合 (Ki > 100 μM)"
    ]
    
    # 初始化结果
    bin_names = []
    bin_counts = []
    bin_rmses = []
    bin_maes = []
    bin_r2s = []
    bin_cis = []  # 一致性指数
    bin_biases = []  # 平均偏差 (预测值 - 实际值)
    
    # 对每个区间计算指标
    for i in range(len(bin_labels)):
        bin_name = bin_labels[i]
        
        # 获取当前区间的样本
        if i == 0:
            mask = (targets < bin_edges[i+1])
        elif i == len(bin_labels) - 1:
            mask = (targets >= bin_edges[i])
        else:
            mask = (targets >= bin_edges[i]) & (targets < bin_edges[i+1])
        
        bin_targets = targets[mask]
        bin_preds = predictions[mask]
        
        # 如果区间内有样本，计算指标
        if len(bin_targets) > 0:
            bin_rmse = np.sqrt(mean_squared_error(bin_targets, bin_preds))
            bin_mae = mean_absolute_error(bin_targets, bin_preds)
            bin_r2 = r2_score(bin_targets, bin_preds) if len(bin_targets) > 1 else float('nan')
            bin_ci = calculate_ci(bin_targets, bin_preds) if len(bin_targets) > 1 else float('nan')
            bin_bias = np.mean(bin_preds - bin_targets)
            bin_count = len(bin_targets)
            
            bin_names.append(bin_name)
            bin_counts.append(bin_count)
            bin_rmses.append(bin_rmse)
            bin_maes.append(bin_mae)
            bin_r2s.append(bin_r2)
            bin_cis.append(bin_ci)
            bin_biases.append(bin_bias)
    
    # 创建DataFrame
    results_df = pd.DataFrame({
        'Affinity Level': bin_names,
        'Count': bin_counts,
        'Percentage': [count / len(targets) * 100 for count in bin_counts],
        'RMSE': bin_rmses,
        'MAE': bin_maes,
        'R2': bin_r2s,
        'CI': bin_cis,
        'Bias': bin_biases
    })
    
    # 保存结果
    results_df.to_csv(os.path.join(output_dir, 'affinity_level_analysis.csv'), index=False)
    
    # 绘制区间RMSE柱状图
    plt.figure(figsize=(14, 6))
    plt.bar(bin_names, bin_rmses, alpha=0.7)
    plt.axhline(y=np.sqrt(mean_squared_error(targets, predictions)), 
                color='r', linestyle='--', label='整体RMSE')
    plt.title('各亲和力等级的RMSE')
    plt.xlabel('亲和力等级')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 特别处理负号问题
    ax = plt.gca()
    for tick in ax.get_yticklabels():
        tick.set_text(tick.get_text().replace('−', '-'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'affinity_level_rmse.png'), dpi=300)
    plt.close()
    
    # 绘制区间MAE柱状图
    plt.figure(figsize=(14, 6))
    plt.bar(bin_names, bin_maes, alpha=0.7)
    plt.axhline(y=mean_absolute_error(targets, predictions), 
                color='r', linestyle='--', label='整体MAE')
    plt.title('各亲和力等级的MAE')
    plt.xlabel('亲和力等级')
    plt.ylabel('MAE')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 特别处理负号问题
    ax = plt.gca()
    for tick in ax.get_yticklabels():
        tick.set_text(tick.get_text().replace('−', '-'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'affinity_level_mae.png'), dpi=300)
    plt.close()
    
    # 绘制区间偏差柱状图
    plt.figure(figsize=(14, 6))
    plt.bar(bin_names, bin_biases, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', label='零偏差线')
    plt.axhline(y=np.mean(predictions - targets), 
                color='r', linestyle='--', label='整体平均偏差')
    plt.title('各亲和力等级的平均偏差 (预测值 - 实际值)')
    plt.xlabel('亲和力等级')
    plt.ylabel('平均偏差')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 特别处理负号问题
    ax = plt.gca()
    for tick in ax.get_yticklabels():
        tick.set_text(tick.get_text().replace('−', '-'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'affinity_level_bias.png'), dpi=300)
    plt.close()
    
    # 绘制样本分布饼图
    plt.figure(figsize=(10, 8))
    plt.pie(bin_counts, labels=bin_names, autopct='%1.1f%%', 
            startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('亲和力等级分布')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'affinity_level_distribution.png'), dpi=300)
    plt.close()
    
    return results_df

def plot_averaging_analysis(predictions, targets, output_dir):
    """分析平均化现象"""
    # 计算目标值与均值的偏差
    target_mean = np.mean(targets)
    target_deviations = targets - target_mean
    
    # 计算预测偏差 (预测值 - 实际值)
    prediction_errors = predictions - targets
    
    # 创建散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(target_deviations, prediction_errors, alpha=0.5, s=20)
    
    # 添加水平线 (y=0)
    plt.axhline(y=0, color='k', linestyle='-', label='零误差线')
    
    # 添加回归线
    z = np.polyfit(target_deviations, prediction_errors, 1)
    p = np.poly1d(z)
    plt.plot(target_deviations, p(target_deviations), 'r-', 
             label=f'趋势线 (y={z[0]:.2f}x+{z[1]:.2f})')
    
    # 添加标题和标签
    plt.title('平均化现象分析\n斜率越接近-1，平均化现象越严重')
    plt.xlabel('目标值偏离均值的程度 (实际值 - 均值)')
    plt.ylabel('预测偏差 (预测值 - 实际值)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 特别处理负号问题
    ax = plt.gca()
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_text(tick.get_text().replace('−', '-'))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'averaging_analysis.png'), dpi=300)
    plt.close()
    
    # 返回平均化系数 (趋势线斜率)
    return z[0]

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置绘图风格和字体
    if HAS_SEABORN:
        sns.set(style="whitegrid")
    
    # 设置中文字体
    set_chinese_font()
    
    # 设置字体大小
    plt.rcParams.update({'font.size': 12})
    
    # 加载测试结果
    results = load_results(args.results_file)
    
    # 提取预测值和实际值
    predictions = np.array(results['predictions'])
    targets = np.array(results['targets'])
    
    # 绘制预测值vs实际值散点图
    plot_predictions_vs_targets(predictions, targets, args.output_dir)
    
    # 绘制残差图
    plot_residuals(predictions, targets, args.output_dir)
    
    # 绘制分布比较
    plot_distribution_comparison(predictions, targets, args.output_dir)
    
    # 按值域范围分析
    range_df = analyze_by_range(predictions, targets, args.num_bins, args.output_dir)
    
    # 按亲和力等级分析
    affinity_df = analyze_by_affinity_level(predictions, targets, args.output_dir)
    
    # 分析平均化现象
    averaging_coefficient = plot_averaging_analysis(predictions, targets, args.output_dir)
    
    # 计算一致性指数
    ci = calculate_ci(targets, predictions)
    
    # 打印分析结果
    print("\n预测性能分析结果:")
    print(f"总样本数: {len(targets)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(targets, predictions)):.4f}")
    print(f"MAE: {mean_absolute_error(targets, predictions):.4f}")
    print(f"R²: {r2_score(targets, predictions):.4f}")
    print(f"CI: {ci:.4f} (一致性指数，越接近1表示排序能力越强)")
    print(f"平均偏差: {np.mean(predictions - targets):.4f}")
    print(f"平均化系数: {averaging_coefficient:.4f} (越接近-1，平均化现象越严重)")
    
    print("\n按值域区间分析:")
    print(range_df.to_string(index=False))
    
    print("\n按亲和力等级分析:")
    print(affinity_df.to_string(index=False))
    
    print(f"\n可视化结果已保存到: {args.output_dir}")
    print("可视化完成！如果图表中的中文仍然显示为方框，请尝试安装中文字体或使用英文标签。")

if __name__ == '__main__':
    main()
