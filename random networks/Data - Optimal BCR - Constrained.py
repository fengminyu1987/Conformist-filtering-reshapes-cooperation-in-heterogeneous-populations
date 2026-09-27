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

# --- 核心算法：最小化从众者的分数规划 ---
def solve_minimal_ST(C_num, C_den, conformist_number, max_iter=100, tol=1e-6):
    n = len(C_num)
    K = n - conformist_number
    
    alpha = np.zeros(n)
    indices = np.random.choice(n, K, replace=False)
    alpha[indices] = 1
    
    curr_numerator = np.dot(alpha, C_num)
    curr_denominator = np.dot(alpha, C_den)
    
    if curr_denominator == 0:
        raise ValueError("Initial denominator is zero.")
        
    lam = curr_numerator / curr_denominator
    
    for iteration in range(max_iter):
        w = C_num - lam * C_den
        sorted_indices = np.argsort(w)
        top_k_indices = sorted_indices[:K]
        
        new_alpha = np.zeros(n)
        new_alpha[top_k_indices] = 1
        min_obj_val = np.sum(w[top_k_indices])
        
        if abs(min_obj_val) < tol:
            break
            
        curr_numerator = np.dot(new_alpha, C_num)
        curr_denominator = np.dot(new_alpha, C_den)
        if curr_denominator <= 1e-9: 
            # print("Warning: Denominator non-positive in iteration.")
            return new_alpha, np.inf
        lam = curr_numerator / curr_denominator
        alpha = new_alpha
    return alpha, lam

# --- 物理量提取 ---
def get_bcr_matrices(G):
    adjm = BCR.adjacency(G)
    degrees = np.sum(adjm, axis=1)
    P = adjm / degrees[:, np.newaxis]
    eta = BCR.solve_eta_matrix_exactly(adjm)
    Ci, Ri = BCR.calculate_CR(P, eta)
    return Ci, Ri

# --- 新增：计算 Collective Influence (CI, l=2) ---
def compute_ci(G, l=2):
    ci = {}
    degrees = dict(G.degree())
    for node in G.nodes():
        frontier = {node}
        visited = {node}
        for _ in range(l):
            next_frontier = set()
            for v in frontier:
                for u in G.neighbors(v):
                    if u not in visited:
                        visited.add(u)
                        next_frontier.add(u)
            frontier = next_frontier
        
        # CI_l(i) = (k_i - 1) * sum_{j in Frontier(i, l)} (k_j - 1)
        ci_val = (degrees[node] - 1) * sum(max(0, degrees[j] - 1) for j in frontier)
        ci[node] = ci_val
    return ci

# --- 新增：给定中心性指标字典，计算对应的合作阈值 ---
def evaluate_heuristic(metric_dict, C_num, C_den, conformist_number):
    n = len(C_num)
    # 按照中心性得分从大到小排序节点
    sorted_nodes = sorted(metric_dict.keys(), key=lambda x: metric_dict[x], reverse=True)
    
    # 选取得分最高的 conformist_number 个节点作为从众者
    conformists = sorted_nodes[:conformist_number]
    
    # 初始化 alpha 向量 (1 表示利益驱动者，0 表示从众者)
    alpha = np.ones(n)
    alpha[conformists] = 0
    
    # 计算当前配置下的阈值 (b/c)*
    return np.dot(alpha, C_num) / np.dot(alpha, C_den)

# --- 主循环 ---
def run_simulation_sequential(net_type, conformist_ratio, total_samples=20000):
    save_dir = ".\\saves"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    file_name = os.path.join(save_dir, f"{net_type}_{conformist_ratio}_Centrality_Compare.csv")
    
    if os.path.exists(file_name):
        os.remove(file_name)

    print(f"开始生成 {net_type} 网络数据 (规模: {total_samples})...")
    
    # 定义将要保存的列名
    columns = [
        'initial_bcr', 'optimal_bcr', 'Degree', 'CI', 
        'K-Shell', 'Betweenness', 'Closeness', 'Eigenvector', 'PageRank'
    ]
    
    for i in tqdm(range(total_samples), desc=f"{net_type} 进度"):
        try:
            # 1. 随机参数生成
            n = np.random.randint(100, 301)
            m = np.random.randint(2, 6) 
            conformist_number = int(n * conformist_ratio)
            
            if net_type == 'BA':
                G = nx.barabasi_albert_graph(n, m)
            elif net_type == 'WS':
                G = nx.connected_watts_strogatz_graph(n, m*2, np.random.uniform(0.1, 0.2))
            elif net_type == 'HK':
                G = nx.powerlaw_cluster_graph(n, m, np.random.uniform(0.1, 0.5))
            
            # 2. 连通性保证
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            
            # 由于子图提取可能会改变节点编号，重置为 0~N-1
            G = nx.convert_node_labels_to_integers(G)
            n = G.number_of_nodes() # 更新实际节点数
            conformist_number = int(n * conformist_ratio) # 更新实际从众者数
            
            # 3. 计算 BCR 矩阵
            C_num, C_den = get_bcr_matrices(G) 
            
            # --- 4. 计算 Optimal 和 Initial 阈值 ---
            lambda_initial = np.sum(C_num) / np.sum(C_den)
            _, lambda_optimal = solve_minimal_ST(C_num, C_den, conformist_number, max_iter=100, tol=1e-6)
            
            # --- 5. 计算各大中心性指标下的阈值 ---
            # 5.1 Degree
            deg_dict = dict(G.degree())
            bcr_deg = evaluate_heuristic(deg_dict, C_num, C_den, conformist_number)
            
            # 5.2 CI (Collective Influence)
            ci_dict = compute_ci(G, l=2)
            bcr_ci = evaluate_heuristic(ci_dict, C_num, C_den, conformist_number)
            
            # 5.3 K-Shell (Core number)
            ks_dict = nx.core_number(G)
            bcr_ks = evaluate_heuristic(ks_dict, C_num, C_den, conformist_number)
            
            # 5.4 Betweenness
            bet_dict = nx.betweenness_centrality(G)
            bcr_bet = evaluate_heuristic(bet_dict, C_num, C_den, conformist_number)
            
            # 5.5 Closeness
            clo_dict = nx.closeness_centrality(G)
            bcr_clo = evaluate_heuristic(clo_dict, C_num, C_den, conformist_number)
            
            # 5.6 Eigenvector (使用 numpy 版本避免迭代不收敛报错)
            eig_dict = nx.eigenvector_centrality_numpy(G)
            bcr_eig = evaluate_heuristic(eig_dict, C_num, C_den, conformist_number)
            
            # 5.7 PageRank
            pr_dict = nx.pagerank(G)
            bcr_pr = evaluate_heuristic(pr_dict, C_num, C_den, conformist_number)
            
            # --- 6. 整理并写入数据 ---
            row_data = [[
                lambda_initial, lambda_optimal, bcr_deg, bcr_ci, 
                bcr_ks, bcr_bet, bcr_clo, bcr_eig, bcr_pr
            ]]
            df_row = pd.DataFrame(row_data, columns=columns)
            
            df_row.to_csv(file_name, mode='a', index=False, header=not os.path.exists(file_name))
            
        except Exception as e:
            # print(f"Error encountered: {e}")
            continue

# --- 执行区 ---
if __name__ == "__main__":
    t1 = time.time()
    
    # 为节省时间，你可以先将 total_samples 调小测试（如 100），确认无误后再跑 20000
    # run_simulation_sequential('BA', 0.2, total_samples=100) 
    
    run_simulation_sequential('BA', 0.1, total_samples=200)
    run_simulation_sequential('WS', 0.1, total_samples=200)
    run_simulation_sequential('HK', 0.1, total_samples=200)
    
    print(f"全部完成！耗时: {(time.time() - t1) / 3600:.4f} Hours")