# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 06:46:54 2026

@author: ziyan
"""

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import BCR

# ==========================================
# 这里放入你已经确定的核心计算函数
# ==========================================

# ==========================================
# 实验参数设置
# ==========================================
N = 100                           # 网络规模
degrees = [4, 6, 8, 10]           # 网络度数 k
proportions = np.round(np.linspace(0.1, 0.9, 9), 1)  # 从众者比例: 0.1, 0.2, ..., 0.9
num_trials = 2000                 # 每个配置下的独立实验次数

# ==========================================
# 主循环生成数据
# ==========================================
def generate_supplementary_data():
    print("开始生成 Random Regular Graphs 数据...")
    
    for k in degrees:
        print(f"\n--- 正在计算度数 k = {k} 的网络 ---")
        
        # 初始化一个字典来存储当前度数下的所有结果，Key为比例
        results_dict = {f"p_{p}": [] for p in proportions}
        
        # 遍历所有设定的从众者比例
        for p in proportions:
            num_conformists = int(np.floor(N * p)) # 向下取整获取从众者数量
            
            # 使用 tqdm 添加进度条，方便观察 1000 次循环的进度
            for _ in tqdm(range(num_trials), desc=f"Proportion {p*100:.0f}%"):
                # 1. 生成独立的随机规则图
                G = nx.random_regular_graph(k, N)
                adj_matrix = nx.to_numpy_array(G)
                
                # 2. 构造从众者向量 alpha
                # 从众者为 0, 利益驱动者为 1
                alpha = np.ones(N)
                
                # 随机抽取 num_conformists 个节点作为从众者
                conformist_indices = random.sample(range(N), num_conformists)
                alpha[conformist_indices] = 0.0
                
                # 3. 计算 (b/c)*
                bc_star = BCR.Get_BC_Ratio(adj_matrix, alpha)
                
                # 将结果记录到字典中
                results_dict[f"p_{p}"].append(bc_star)
                
        # 4. 将当前度数 k 的结果保存为 CSV 文件
        # 每列是一个比例，每列有 1000 行
        df = pd.DataFrame(results_dict)
        filename = f".\\saves\\RRG_bc_ratio_k{k}.csv"
        df.to_csv(filename, index=False)
        print(f"度数 k={k} 的数据已保存至 {filename}")

    print("\n所有数据生成完毕！")

# 运行主程序
if __name__ == "__main__":
    generate_supplementary_data()