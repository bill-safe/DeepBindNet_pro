# 导入必要的库
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
    parser = argparse.ArgumentParser(description='DeepBindNet训练可视化脚本')
    parser.add_argument('--log_dir', type=str, default='outputs_optimized/logs',
                        help='TensorBoard日志目录')
    parser.add_argument('--output_dir', type=str, default='outputs_optimized/visualizations',
                        help='可视化输出目录')
    return parser.parse_args()

def load_tensorboard_data(log_dir):
    """从TensorBoard日志加载数据"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # 获取所有标量标签
    tags = event_acc.Tags()['scalars']
    
    # 加载数据
    data = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        data[tag] = {
            'step': [event.step for event in events],
            'value': [event.value for event in events]
        }
    
    return data

def plot_metrics(data, output_dir):
    """绘制训练指标图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置绘图风格和字体
    if HAS_SEABORN:
        sns.set(style="whitegrid")
    
    # 设置中文字体
    set_chinese_font()
    
    # 设置字体大小
    plt.rcParams.update({'font.size': 12})
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    if 'Loss/train' in data:
        plt.plot(data['Loss/train']['step'], data['Loss/train']['value'], label='训练损失', marker='o', markersize=4)
    if 'Loss/val' in data:
        plt.plot(data['Loss/val']['step'], data['Loss/val']['value'], label='验证损失', marker='s', markersize=4)
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    
    # 绘制RMSE曲线
    plt.figure(figsize=(12, 6))
    if 'RMSE/train' in data:
        plt.plot(data['RMSE/train']['step'], data['RMSE/train']['value'], label='训练RMSE', marker='o', markersize=4)
    if 'RMSE/val' in data:
        plt.plot(data['RMSE/val']['step'], data['RMSE/val']['value'], label='验证RMSE', marker='s', markersize=4)
    plt.xlabel('轮次')
    plt.ylabel('RMSE')
    plt.title('训练和验证RMSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_curve.png'), dpi=300)
    plt.close()
    
    # 绘制R2曲线
    plt.figure(figsize=(12, 6))
    if 'R2/train' in data:
        plt.plot(data['R2/train']['step'], data['R2/train']['value'], label='训练R2', marker='o', markersize=4)
    if 'R2/val' in data:
        plt.plot(data['R2/val']['step'], data['R2/val']['value'], label='验证R2', marker='s', markersize=4)
    plt.xlabel('轮次')
    plt.ylabel('R2分数')
    plt.title('训练和验证R2分数')
    plt.legend()
    plt.grid(True)
    
    # 特别处理R2图表的负号问题
    ax = plt.gca()
    # 使用ASCII负号代替Unicode负号
    for tick in ax.get_yticklabels():
        tick.set_text(tick.get_text().replace('−', '-'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_curve.png'), dpi=300)
    plt.close()
    
    # 绘制所有指标的组合图
    plt.figure(figsize=(18, 12))
    
    # 损失子图
    plt.subplot(2, 2, 1)
    if 'Loss/train' in data:
        plt.plot(data['Loss/train']['step'], data['Loss/train']['value'], label='训练损失', marker='o', markersize=3)
    if 'Loss/val' in data:
        plt.plot(data['Loss/val']['step'], data['Loss/val']['value'], label='验证损失', marker='s', markersize=3)
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    
    # RMSE子图
    plt.subplot(2, 2, 2)
    if 'RMSE/train' in data:
        plt.plot(data['RMSE/train']['step'], data['RMSE/train']['value'], label='训练RMSE', marker='o', markersize=3)
    if 'RMSE/val' in data:
        plt.plot(data['RMSE/val']['step'], data['RMSE/val']['value'], label='验证RMSE', marker='s', markersize=3)
    plt.xlabel('轮次')
    plt.ylabel('RMSE')
    plt.title('训练和验证RMSE')
    plt.legend()
    plt.grid(True)
    
    # R2子图
    plt.subplot(2, 2, 3)
    if 'R2/train' in data:
        plt.plot(data['R2/train']['step'], data['R2/train']['value'], label='训练R2', marker='o', markersize=3)
    if 'R2/val' in data:
        plt.plot(data['R2/val']['step'], data['R2/val']['value'], label='验证R2', marker='s', markersize=3)
    plt.xlabel('轮次')
    plt.ylabel('R2分数')
    plt.title('训练和验证R2分数')
    plt.legend()
    plt.grid(True)
    
    # 特别处理R2子图的负号问题
    ax = plt.gca()
    # 使用ASCII负号代替Unicode负号
    for tick in ax.get_yticklabels():
        tick.set_text(tick.get_text().replace('−', '-'))
    
    # 学习率子图
    plt.subplot(2, 2, 4)
    if 'lr' in data:
        plt.plot(data['lr']['step'], data['lr']['value'], label='学习率', marker='o', markersize=3)
        plt.xlabel('轮次')
        plt.ylabel('学习率')
        plt.title('学习率变化')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, '学习率数据不可用', horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics.png'), dpi=300)
    plt.close()
    
    print(f"可视化图表已保存到 {output_dir}")

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载TensorBoard数据
    data = load_tensorboard_data(args.log_dir)
    
    # 绘制指标图
    plot_metrics(data, args.output_dir)
    
    # 打印完成信息
    print("可视化完成！如果图表中的中文仍然显示为方框，请尝试安装中文字体或使用英文标签。")

if __name__ == '__main__':
    main()
