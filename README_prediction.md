# DeepBindNet 预测脚本使用指南

本文档提供了如何使用训练好的 DeepBindNet 模型进行蛋白质-小分子结合亲和力预测的说明。

## 环境要求

确保您已安装以下依赖项：

```bash
pip install torch torch-geometric rdkit matplotlib pandas tqdm numpy
```

## 文件说明

- `predict.py`: 单个分子-蛋白质对预测脚本
- `batch_predict.py`: 批量预测脚本
- `sample_input.csv`: 批量预测的示例输入文件

## 单个预测

使用 `predict.py` 脚本对单个分子-蛋白质对进行预测：

```bash
python predict.py --smiles "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC(=O)C4=CC=C(C=C4)F" --protein_sequence "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNPGILPIQPALTPRKRATVQPPPQAAGSSNQPGLQNGLNVQKRKQELAGTLTAELEELKQAVPKEAGPGSQPQ"
```

### 参数说明

- `--smiles`: 分子的 SMILES 字符串
- `--protein_sequence`: 蛋白质序列
- `--model_type`: 模型类型，可选 'standard' 或 'gated'（默认为 'gated'）
- `--model_dir`: 模型目录，如果为 None 则使用默认目录
- `--device`: 预测设备，'cuda' 或 'cpu'
- `--visualize`: 是否可视化注意力权重
- `--output_dir`: 输出目录

## 批量预测

使用 `batch_predict.py` 脚本对多个分子-蛋白质对进行批量预测：

```bash
python batch_predict.py --input_csv sample_input.csv
```

### 参数说明

- `--input_csv`: 输入 CSV 文件，必须包含 'smiles' 和 'protein_sequence' 列
- `--model_type`: 模型类型，可选 'standard' 或 'gated'（默认为 'gated'）
- `--model_dir`: 模型目录，如果为 None 则使用默认目录
- `--device`: 预测设备，'cuda' 或 'cpu'
- `--batch_size`: 批处理大小（默认为 32）
- `--output_dir`: 输出目录

### 输入 CSV 格式

输入 CSV 文件必须包含以下列：

- `smiles`: 分子的 SMILES 字符串
- `protein_sequence`: 蛋白质序列

示例：

```csv
smiles,protein_sequence
CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC(=O)C4=CC=C(C=C4)F,MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNPGILPIQPALTPRKRATVQPPPQAAGSSNQPGLQNGLNVQKRKQELAGTLTAELEELKQAVPKEAGPGSQPQ
```

## 输出说明

### 单个预测输出

单个预测会输出以下内容：

- 终端显示预测的 KIBA 分数、结合亲和力强度和药理活性可能性
- 在输出目录中保存 `prediction_result.pkl` 文件
- 如果启用可视化，还会生成 `attention_weights.png` 文件

### 批量预测输出

批量预测会输出以下内容：

- `batch_predictions.csv`: 包含所有预测结果的 CSV 文件
- `batch_predictions.pkl`: 包含所有预测结果的 pickle 文件
- `kiba_score_distribution.png`: KIBA 分数分布直方图
- `binding_strength_distribution.png`: 结合亲和力分类饼图
- `statistics.pkl`: 包含统计信息的 pickle 文件

## KIBA 分数解释

KIBA 分数是一种综合了 Ki、Kd 和 IC50 的标准化指标，用于衡量蛋白质-小分子结合亲和力。分数越高，表示结合亲和力越强。

- \> 12.0: 极强结合亲和力，非常可能有药理活性
- 8.0 - 12.0: 强结合亲和力，很可能有药理活性
- 6.0 - 8.0: 中等结合亲和力，可能有药理活性
- 4.0 - 6.0: 弱结合亲和力，药理活性有限
- < 4.0: 极弱结合亲和力，药理活性可能性低

## 注意事项

1. 确保模型文件（`best_model.pt`）位于正确的目录中
2. 对于大型蛋白质序列，预测可能需要较长时间
3. 使用 GPU 可以显著加速预测过程
4. SMILES 字符串必须有效，否则会导致错误
