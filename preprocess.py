# 导入必要的库
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data, Batch
import esm
import pickle
from gin import MolecularGraphProcessor
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    数据预处理类，负责:
    1. 加载KIBA数据集
    2. 使用ESM模型提取蛋白质序列特征
    3. 使用RDKit将SMILES转化为分子图
    4. 保存处理好的数据
    """
    def __init__(self, data_path, output_dir, esm_model_path=None):
        """
        初始化数据预处理器
        
        参数:
        - data_path: KIBA数据集路径
        - output_dir: 输出目录
        - esm_model_path: ESM预训练模型路径，如果为None则自动下载
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.esm_model_path = esm_model_path
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化分子处理器
        self.mol_processor = MolecularGraphProcessor()
        
        # 加载ESM模型
        self._load_esm_model()
    
    def _load_esm_model(self):
        """加载ESM预训练模型"""
        print("加载ESM预训练模型...")
        
        # 检查CUDA是否可用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 创建模型目录
        model_dir = os.path.join(os.getcwd(), "model")
        os.makedirs(model_dir, exist_ok=True)
        
        if self.esm_model_path and os.path.exists(self.esm_model_path):
            # 从本地加载模型
            print(f"从本地加载ESM模型: {self.esm_model_path}")
            self.esm_model, self.esm_alphabet = torch.hub.load_state_dict_from_url(
                f"file://{self.esm_model_path}", map_location=device
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
        
        # 将模型移动到GPU（如果可用）
        self.esm_model = self.esm_model.to(device)
        self.esm_model = self.esm_model.eval()  # 设置为评估模式
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        self.device = device  # 保存设备以便后续使用
        print("ESM模型加载完成")
    
    def load_data(self):
        """加载KIBA数据集"""
        print(f"加载数据集: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"数据集加载完成，共 {len(self.df)} 条记录")
        
        # 打印列名以便调试
        print(f"数据集列名: {list(self.df.columns)}")
        
        # 根据实际列名创建新的列
        # 基于输出的列名: ['CHEMBLID', 'ProteinID', 'compound_iso_smiles', 'target_sequence', 'Ki  Kd and IC50  (KIBA Score)']
        if 'compound_iso_smiles' in self.df.columns and 'smiles' not in self.df.columns:
            self.df['smiles'] = self.df['compound_iso_smiles']
            print("已将'compound_iso_smiles'映射为'smiles'")
            
        if 'target_sequence' in self.df.columns and 'protein_sequence' not in self.df.columns:
            self.df['protein_sequence'] = self.df['target_sequence']
            print("已将'target_sequence'映射为'protein_sequence'")
            
        # 处理KIBA分数列，注意列名中可能有空格
        kiba_column = None
        for col in self.df.columns:
            if 'KIBA' in col or 'kiba' in col:
                kiba_column = col
                break
        
        if kiba_column and 'kiba_score' not in self.df.columns:
            self.df['kiba_score'] = self.df[kiba_column]
            print(f"已将'{kiba_column}'映射为'kiba_score'")
        
        # 再次打印列名以确认添加的新列
        print(f"处理后的列名: {list(self.df.columns)}")
        
        # 检查必要的列是否存在
        required_columns = ['protein_sequence', 'smiles', 'kiba_score']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"数据集缺少必要的列: {missing_columns}")
        
        return self.df
    
    def extract_protein_features(self, sequences, batch_size=4):
        """
        使用ESM模型提取蛋白质序列特征
        
        参数:
        - sequences: 蛋白质序列列表
        - batch_size: 批处理大小
        
        返回:
        - 特征字典，键为序列，值为特征向量
        """
        print("提取蛋白质序列特征...")
        features = {}
        
        # 准备批次
        batches = []
        current_batch = []
        
        for i, seq in enumerate(sequences):
            if seq in features:  # 跳过已处理的序列
                continue
                
            current_batch.append((f"protein_{i}", seq))
            
            if len(current_batch) == batch_size:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:  # 添加最后一个不完整的批次
            batches.append(current_batch)
        
        # 处理每个批次
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(batches, desc="处理蛋白质批次")):
                batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(batch)
                
                # 将tokens移动到GPU
                batch_tokens = batch_tokens.to(self.device)
                
                # 使用ESM模型提取特征
                results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
                
                # 保存每个序列的特征
                for i, (_, seq) in enumerate(batch):
                    # 使用平均池化获取序列表示
                    seq_rep = token_representations[i, 1:len(seq)+1].mean(0)
                    features[seq] = seq_rep.cpu().numpy()
        
        print(f"蛋白质特征提取完成，共 {len(features)} 个唯一序列")
        return features
    
    def process_molecules(self, smiles_list):
        """
        处理分子SMILES字符串列表
        
        参数:
        - smiles_list: SMILES字符串列表
        
        返回:
        - 分子图字典，键为SMILES，值为分子图对象
        """
        print("处理分子SMILES...")
        mol_graphs = {}
        
        for smiles in tqdm(smiles_list, desc="处理分子"):
            if smiles in mol_graphs:  # 跳过已处理的分子
                continue
                
            try:
                # 使用MolecularGraphProcessor处理分子
                mol_graph = self.mol_processor.process_molecule(smiles)
                mol_graphs[smiles] = mol_graph
            except Exception as e:
                print(f"处理分子出错 ({smiles}): {e}")
                # 对于无效的SMILES，创建一个空图
                mol_graphs[smiles] = None
        
        print(f"分子处理完成，共 {len(mol_graphs)} 个唯一分子")
        return mol_graphs
    
    def _normalize_kiba_scores(self, kiba_scores):
        """标准化KIBA分数"""
        print("标准化KIBA分数...")
        scaler = StandardScaler()
        normalized_scores = scaler.fit_transform(kiba_scores.reshape(-1, 1))
        
        # 保存scaler参数用于后续反标准化
        scaler_params = {
            'mean': scaler.mean_,
            'scale': scaler.scale_
        }
        
        # 将scaler参数保存到文件
        scaler_path = os.path.join(self.output_dir, 'scaler_params.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_params, f)
        print(f"Scaler参数已保存至: {scaler_path}")
        
        return normalized_scores.squeeze()

    def preprocess(self):
        """执行完整的预处理流程"""
        # 加载数据
        self.load_data()
        
        # 标准化KIBA分数
        self.df['kiba_score'] = self._normalize_kiba_scores(self.df['kiba_score'].values)
        
        # 获取唯一的蛋白质序列和SMILES
        unique_proteins = self.df['protein_sequence'].unique()
        unique_smiles = self.df['smiles'].unique()
        
        print(f"唯一蛋白质序列数量: {len(unique_proteins)}")
        print(f"唯一SMILES数量: {len(unique_smiles)}")
        
        # 提取蛋白质特征
        protein_features = self.extract_protein_features(unique_proteins)
        
        # 处理分子
        mol_graphs = self.process_molecules(unique_smiles)
        
        # 创建处理后的数据集
        processed_data = []
        
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="创建最终数据集"):
            protein_seq = row['protein_sequence']
            smiles = row['smiles']
            kiba_score = row['kiba_score']
            
            # 获取蛋白质特征和分子图
            protein_feature = protein_features.get(protein_seq)
            mol_graph = mol_graphs.get(smiles)
            
            # 只添加有效的数据
            if protein_feature is not None and mol_graph is not None:
                processed_data.append({
                    'protein_sequence': protein_seq,
                    'protein_feature': protein_feature,
                    'smiles': smiles,
                    'mol_graph': mol_graph,
                    'kiba_score': kiba_score
                })
        
        print(f"最终处理后的数据集大小: {len(processed_data)}")
        
        # 保存处理后的数据
        output_path = os.path.join(self.output_dir, 'processed_data.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"数据已保存至: {output_path}")
        
        # 数据划分
        self._split_data(processed_data)
        
        return processed_data
    
    def _split_data(self, processed_data):
        """
        按照7:2:1的比例划分数据集
        
        参数:
        - processed_data: 处理后的数据列表
        """
        print("划分数据集...")
        
        # 随机打乱数据
        np.random.seed(101113)
        indices = np.random.permutation(len(processed_data))
        
        # 计算划分点
        train_end = int(0.7 * len(processed_data))
        val_end = int(0.9 * len(processed_data))
        
        # 划分数据
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # 创建数据集
        train_data = [processed_data[i] for i in train_indices]
        val_data = [processed_data[i] for i in val_indices]
        test_data = [processed_data[i] for i in test_indices]
        
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}")
        print(f"测试集大小: {len(test_data)}")
        
        # 保存划分后的数据集
        with open(os.path.join(self.output_dir, 'train_data.pkl'), 'wb') as f:
            pickle.dump(train_data, f)
        
        with open(os.path.join(self.output_dir, 'val_data.pkl'), 'wb') as f:
            pickle.dump(val_data, f)
        
        with open(os.path.join(self.output_dir, 'test_data.pkl'), 'wb') as f:
            pickle.dump(test_data, f)
        
        print("数据集划分完成")

# 使用示例
if __name__ == "__main__":
    # 设置路径
    data_path = "data/KIBA.csv"
    output_dir = "data/processed"
    esm_model_path = None  # 设置为None以触发自动下载
    
    # 创建预处理器
    preprocessor = DataPreprocessor(
        data_path=data_path,
        output_dir=output_dir,
        esm_model_path=esm_model_path
    )
    
    # 执行预处理
    processed_data = preprocessor.preprocess()
    
    print("预处理完成!")
