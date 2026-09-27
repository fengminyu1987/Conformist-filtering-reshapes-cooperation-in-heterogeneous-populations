# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:43:45 2026

@author: ziyan
"""

import BCR
import networkx as nx
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import warnings

# 忽略全 NaN 切片时的 numpy 警告 (防止控制台被刷屏)
warnings.filterwarnings('ignore', r'All-NaN slice encountered')

# --- 基础设置 (需与你跑前置数据时保持绝对一致) ---
save_dir = ".\\saves"
rev = False      
selfloop = False   
GNAME = 'power-eris1176'  
Graphs = ['petster-hamster-household','arenas-email', 'web-polblogs','bio-CE-LC','power-eris1176']

# 确定你要修改的原始 CSV 文件路径
if rev == True:
    if selfloop == False:
        csv_filename = os.path.join(save_dir, f"{GNAME}_bcr.csv")
    elif selfloop == True:
        csv_filename = os.path.join(save_dir, f"{GNAME}_bcr_self.csv")
elif rev == False:
    csv_filename = os.path.join(save_dir, f"{GNAME}_bcr_asc.csv")

if not os.path.exists(csv_filename):
    raise FileNotFoundError(f"找不到原始文件 {csv_filename}！请先运行之前的中心性代码生成它。")

# --- 1. 读取网络并计算基础物理量 ---
print(f"Loading Graph {GNAME} and computing baseline matrices...")
G = BCR.open_graph(GNAME, selfloop)
G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute='old_label')
N = G.number_of_nodes()

adjm = BCR.adjacency(G)
degrees = np.sum(adjm, axis=1)
P = adjm / degrees[:, np.newaxis]

eta = BCR.solve_eta_matrix_exactly(adjm)
Ci, Ri = BCR.calculate_CR(P, eta)

# --- 2. 随机 Baseline 独立抽样 (中位数+分位数版) ---
trials = 10000  # 注意：10000次抽样非常严谨，但在大网络上可能需要几分钟跑完
random_medians = []
random_q_lowers = []
random_q_uppers = []

print(f"Running Random Baseline Simulation ({trials} trials per step)...")
# 使用 tqdm 进度条
for k in tqdm(range(N), desc="Random Baseline Progress"):
    if k == 0:
        # 0 个从众者的情况是确定的
        val = np.sum(Ci) / np.sum(Ri)
        random_medians.append(val)
        random_q_lowers.append(val)
        random_q_uppers.append(val)
        continue
        
    temp_vals = []
    for _ in range(trials):
        # 核心：利用 numpy 极速不重复抽样 k 个节点
        conformists = np.random.choice(N, k, replace=False)
        
        alpha = np.ones(N)
        alpha[conformists] = 0
        
        num = np.dot(alpha, Ci)
        den = np.dot(alpha, Ri)
        
        if den <= 1e-9:
            temp_vals.append(np.nan)
        else:
            temp_vals.append(num / den)
            
    # 计算中位数(50%)，以及 5% 和 95% 分位数
    # 如果该步全为 NaN，nanpercentile 会安全地返回 NaN
    median_val = np.nanpercentile(temp_vals, 50)
    q_lower = np.nanpercentile(temp_vals, 5)
    q_upper = np.nanpercentile(temp_vals, 95)
    
    random_medians.append(median_val)
    random_q_lowers.append(q_lower)
    random_q_uppers.append(q_upper)

# --- 3. 读取原 CSV 并无缝追加列 ---
print(f"Merging results into {csv_filename}...")
df = pd.read_csv(csv_filename)

# 自动清理历史遗留废弃列（如果有的话）
cols_to_drop = ['Random_Mean', 'Random_Std']
for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"Cleaned up old column: {col}")

# 确保行数对齐 (防呆设计)
if len(df) != len(random_medians):
    print("Warning: CSV 行数与当前计算节点数 N 不一致，跳过合并，单独保存为 random_baseline.csv")
    df_random = pd.DataFrame({
        "fraction_conformists": [i/N for i in range(N)],
        "Random_Median": random_medians,
        "Random_Q_Lower": random_q_lowers,
        "Random_Q_Upper": random_q_uppers
    })
    df_random.to_csv(os.path.join(save_dir, f"{GNAME}_random_baseline.csv"), index=False)
else:
    # 覆盖或新增中位数与分位数列
    df["Random_Median"] = random_medians
    df["Random_Q_Lower"] = random_q_lowers
    df["Random_Q_Upper"] = random_q_uppers
    
    # 重新保存覆盖原文件
    df.to_csv(csv_filename, index=False, float_format='%.8f')
    print(f"Success! Random baseline (Median + Percentiles) added to {csv_filename}")