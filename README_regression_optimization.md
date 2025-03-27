# DeepBindNet 回归模型优化方案

本文档提供了针对 DeepBindNet 回归模型的优化方案实施指南，主要解决预测"平均化"现象，提升极端值预测能力。

## 目录

- [背景说明](#背景说明)
- [优化方案](#优化方案)
- [实施步骤](#实施步骤)
- [使用指南](#使用指南)
- [结果分析](#结果分析)
- [进阶优化](#进阶优化)

## 背景说明

DeepBindNet 模型用于预测蛋白质-小分子之间的结合能力，任务为回归问题。当前模型存在预测"平均化"现象，即预测值趋向于分布中心，对极端值的预测能力较弱。本优化方案旨在缓解这一问题，提升模型的泛化能力和精度。

## 优化方案

本实施方案主要包含两个核心优化策略：

1. **替代损失函数**：提供 Huber Loss 和 Log-Cosh Loss 两种对极端值不敏感的损失函数，减轻平均化现象。

2. **样本加权机制**：根据目标值偏离均值的程度对样本进行加权，使模型更加关注分布两端的样本。

此外，还提供了预测结果可视化和分析工具，帮助评估优化效果。

## 实施步骤

### 1. 文件说明

本优化方案包含以下新增文件：

- `loss_functions.py`: 实现了多种损失函数和样本加权机制
- `train_optimized.py`: 优化版训练脚本，支持选择不同损失函数和样本加权
- `visualize_predictions.py`: 预测结果可视化和分析工具，特别关注平均化现象
- `README_regression_optimization.md`: 本文档，提供使用指南

### 2. 环境准备

确保已安装所有必要的依赖：

```bash
pip install torch torch-geometric scikit-learn matplotlib seaborn pandas scipy
```

## 使用指南

### 1. 使用优化版训练脚本

优化版训练脚本支持多种损失函数和样本加权选项：

```bash
python train_optimized.py \
    --data_dir data/processed \
    --output_dir outputs/optimized_huber_weighted \
    --loss_type huber \
    --huber_delta 1.0 \
    --use_weighted_loss \
    --weight_type abs_diff \
    --weight_alpha 1.0 \
    --batch_size 128 \
    --num_epochs 200 \
    --lr 1e-4
```

主要参数说明：

- `--loss_type`: 损失函数类型，可选 `mse`、`huber`、`log_cosh`
- `--huber_delta`: Huber 损失的 delta 参数，控制对异常值的敏感度
- `--use_weighted_loss`: 是否使用样本加权
- `--weight_type`: 权重计算方式，可选 `abs_diff`、`squared_diff`、`exp_diff`
- `--weight_alpha`: 权重强度系数，值越大权重差异越明显

### 2. 可视化和分析预测结果

训练完成后，使用可视化脚本分析预测结果：

```bash
python visualize_predictions.py \
    --results_file outputs\1\test_results.pkl \
    --output_dir outputs/optimized_huber_weighted/visualizations \
    --num_bins 5
```

该脚本会生成多种可视化图表，包括：

- 预测值 vs 实际值散点图
- 残差分析图
- 预测值与实际值分布比较
- 各值域区间的性能指标
- 平均化现象分析图

### 3. 推荐的优化配置

以下是几种推荐的优化配置，可以根据具体情况选择：

#### 配置 1: Huber Loss + 绝对差异加权

```bash
python train_optimized.py \
    --loss_type huber \
    --huber_delta 1.0 \
    --use_weighted_loss \
    --weight_type abs_diff \
    --weight_alpha 1.0
```

适用场景：数据中存在少量异常值，但需要关注分布两端的样本。

#### 配置 2: Log-Cosh Loss + 指数差异加权

```bash
python train_optimized.py \
    --loss_type log_cosh \
    --use_weighted_loss \
    --weight_type exp_diff \
    --weight_alpha 0.5
```

适用场景：需要平滑处理异常值，同时显著提升对极端值的预测能力。

#### 配置 3: MSE + 平方差异加权

```bash
python train_optimized.py \
    --loss_type mse \
    --use_weighted_loss \
    --weight_type squared_diff \
    --weight_alpha 0.8
```

适用场景：数据分布较为均匀，仅需轻微调整对分布两端的关注度。

## 结果分析

### 评估平均化现象

可视化脚本中的"平均化现象分析图"是评估优化效果的关键指标。该图展示了目标值偏离均值的程度与预测偏差之间的关系：

- **平均化系数**：趋势线斜率，越接近 -1 表示平均化现象越严重
- **理想情况**：斜率接近 0，表示预测偏差与目标值偏离均值无关

### 分区间性能分析

"各值域区间的性能指标"可以帮助评估模型在不同值域范围的表现：

- 对比各区间的 RMSE、MAE 和偏差
- 关注极端值区间（最高和最低区间）的性能
- 理想情况下，各区间的性能指标应该相近

## 进阶优化

本实施方案仅包含了优化计划中的前两项策略。未来可以考虑实施以下进阶优化：

1. **预测区间输出**：修改模型结构，使其输出均值和标准差，实现不确定性估计

2. **数据增强与验证集构造**：使用分层采样和 SMOGN 回归过采样技术增强低频样本

这些进阶优化需要更多的模型结构修改和数据处理工作，可以在基础优化效果验证后再考虑实施。

## 参考文献

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
2. Branco, P., Torgo, L., & Ribeiro, R. P. (2017). SMOGN: a Pre-processing Approach for Imbalanced Regression.
3. Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
