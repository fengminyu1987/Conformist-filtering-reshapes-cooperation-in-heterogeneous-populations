# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 21:21:09 2026

@author: ziyan
"""
import numpy as np
import networkx as nx
import pandas as pd
import os
import BCR
import time
from tqdm import tqdm

# --- 核心算法：最小化从众者的无约束分数规划 ---
def solve_min_conformists(C_num, C_den, epsilon=1e-11, tol=1e-8):
    n = len(C_num)
    alpha = np.ones(n) # 初始全是利益驱动者
    
    num = np.dot(alpha, C_num)
    den = np.dot(alpha, C_den)
    if den <= 0: return alpha, np.inf
    lam = num / den
    
    for _ in range(50):
        w = C_num - lam * C_den
        new_alpha = (w < epsilon).astype(float) # 尽量保持 alpha=1 (非从众)
        
        if np.sum(new_alpha) == 0:
            new_alpha[np.argmin(w)] = 1.0
            
        n_num = np.dot(new_alpha, C_num)
        n_den = np.dot(new_alpha, C_den)
        if n_den <= 0: break
        n_lam = n_num / n_den
        
        if abs(n_lam - lam) < tol:
            return new_alpha, n_lam
        alpha, lam = new_alpha, n_lam
    return alpha, lam

# --- 物理量提取 ---
def get_bcr_matrices(G):
    adjm = BCR.adjacency(G)
    degrees = np.sum(adjm, axis=1)
    P = adjm / degrees[:, np.newaxis]
    eta = BCR.solve_eta_matrix_exactly(adjm)
    Ci, Ri = BCR.calculate_CR(P, eta)
    return Ci, Ri

def process_single_network(args):
    net_type, n, p_m, extra_param = args
    if net_type == 'BA':
        G = nx.barabasi_albert_graph(n, p_m)
    elif net_type == 'WS':
        G = nx.connected_watts_strogatz_graph(n, p_m, extra_param)
    elif net_type == 'HK':
        G = nx.powerlaw_cluster_graph(n, p_m, extra_param)
    
    # 1. 计算原始阈值 (alpha = all ones)
    C_num, C_den = get_bcr_matrices(G) # 需要根据你的模型实现
    lambda_initial = np.sum(C_num) / np.sum(C_den)
    
    # 2. 运行优化算法
    best_alpha, lambda_optimal = solve_min_conformists(C_num, C_den)
    
    # 3. 计算最小从众者比例 (alpha=0的数量 / N)
    min_rho = (len(best_alpha) - np.sum(best_alpha)) / len(best_alpha)
    
    return [lambda_initial, lambda_optimal, min_rho]

def run_simulation_sequential(net_type, total_samples=20000):
    file_name = f".\\saves\\{net_type}.csv"
    
    # 如果文件已存在，先删除，防止重复追加旧数据
    if os.path.exists(file_name):
        os.remove(file_name)

    print(f"开始生成 {net_type} 网络数据...")
    
    for i in tqdm(range(total_samples)):
        try:
            # 1. 随机参数生成
            n = np.random.randint(100, 301)
            m = np.random.randint(2, 6) 
            
            if net_type == 'BA':
                G = nx.barabasi_albert_graph(n, m)
            elif net_type == 'WS':
                G = nx.connected_watts_strogatz_graph(n, m*2, np.random.uniform(0.1, 0.2))
            elif net_type == 'HK':
                G = nx.powerlaw_cluster_graph(n, m, np.random.uniform(0.1, 0.5))
            
            # 2. 连通性保证（非常重要，不连通会导致 BCR 逻辑失效）
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            
            # 3. 计算 BCR 矩阵 (这里请换成你真实的矩阵计算函数)
            C_num, C_den = get_bcr_matrices(G) 
            # 示例占位：
            #C_num = np.random.rand(len(G)) 
            #C_den = np.random.rand(len(G))

            # 4. 计算指标
            lambda_initial = np.sum(C_num) / np.sum(C_den)
            best_alpha, lambda_optimal = solve_min_conformists(C_num, C_den)
            min_rho = (len(best_alpha) - np.sum(best_alpha)) / len(best_alpha)
            
            # 5. 构造单行数据并立即写入 CSV
            df_row = pd.DataFrame([[lambda_initial, lambda_optimal, min_rho]], 
                                  columns=['initial_bcr', 'optimal_bcr', 'min_rho'])
            
            # header 仅在第一次写入时添加
            df_row.to_csv(file_name, mode='a', index=False, header=not os.path.exists(file_name))
            
        except Exception as e:
            # 捕捉异常（如不收敛等），跳过该样本，保证程序不断掉
            continue
t1=time.time()
run_simulation_sequential('BA',total_samples=20000)
run_simulation_sequential('WS',total_samples=20000)
run_simulation_sequential('HK',total_samples=20000)
print(time.time()-t1)