# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 18:26:04 2026

@author: ziyan
"""

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import networkx as nx
import BCR
import matplotlib.pyplot as plt
def solve_minimal_ST(C_num, C_den, conformist_number, max_iter=100, tol=1e-6):
    """
    使用 Dinkelbach 算法优化 (b/c)* = (alpha @ C_num) / (alpha @ C_den)
    约束: sum(alpha) == K
    """
    n = len(C_num)
    K = n-conformist_number
    
    # 1. 初始解 (例如随机选择 K 个)
    alpha = np.zeros(n)
    indices = np.random.choice(n, K, replace=False)
    alpha[indices] = 1
    
    # 计算初始 lambda
    curr_numerator = np.dot(alpha, C_num)
    curr_denominator = np.dot(alpha, C_den)
    
    if curr_denominator == 0:
        raise ValueError("Initial denominator is zero.")
        
    lam = curr_numerator / curr_denominator
    
    for iteration in range(max_iter):
        # 2. 构造权重 w = Num - lambda * Den
        w = C_num - lam * C_den
        
        # 3. 求解线性子问题 (排序选最小的 K 个)
        # argsort 返回的是从小到大的索引
        sorted_indices = np.argsort(w)
        top_k_indices = sorted_indices[:K]
        
        # 构建新的 alpha
        new_alpha = np.zeros(n)
        new_alpha[top_k_indices] = 1
        
        # 4. 计算当前的目标函数值 F(lambda)
        min_obj_val = np.sum(w[top_k_indices])
        
        # 5. 检查收敛
        if abs(min_obj_val) < tol:
            print(f"Converged at iteration {iteration}")
            break
            
        # 6. 更新 lambda
        curr_numerator = np.dot(new_alpha, C_num)
        curr_denominator = np.dot(new_alpha, C_den)
        if curr_denominator <= 1e-9: # 设定一个极小的阈值防止除零
            # 如果当前最优解导致分母非正，说明 lambda 估计过高或者无可行解
            # 在这种物理情境下，通常意味着我们需要“回退”或者终止
            # 但对于"最小化阈值"问题，通常不会走到这一步，除非整个系统都很糟糕
            print("Warning: Denominator non-positive in iteration.")
            
            # 补救措施：如果你必须选 K 个，但选出来的导致分母为负，
            # 物理上意味着阈值是无穷大。
            return new_alpha, np.inf
        lam = curr_numerator / curr_denominator
        alpha = new_alpha
    return alpha, lam
def solve_mininal_unconstrained_with_sparsity(C_num, C_den, max_iter=100, tol=1e-8, epsilon=1e-11):
    """
    带有稀疏约束的 Dinkelbach 算法。
    在保证阈值最小的同时，尽量减少从众者的数量。
    """
    n = len(C_num)
    alpha = (C_den > 0).astype(float)
    
    # 初始 lambda
    curr_num = np.dot(alpha, C_num)
    curr_den = np.dot(alpha, C_den)
    lam = curr_num / curr_den
    
    for iteration in range(max_iter):
        # --- 2. 构造权重 w ---
        w = C_num - lam * C_den
        
        # --- 3. 求解子问题 (引入 epsilon 惩罚数量) ---
        # 只有当边际贡献显著小于 0 时才选
        # 这会自动过滤掉那些对降低阈值没有贡献（w=0）的节点
        new_alpha = (w < -epsilon).astype(float) 
        
        # --- 4. 防退化处理 ---
        # 如果所有节点的贡献都大于 -epsilon，我们必须至少选一个最优秀的点
        if np.sum(new_alpha) == 0:
            best_idx = np.argmin(w)
            new_alpha[best_idx] = 1.0
            
        # --- 后续逻辑不变 ---
        next_num = np.dot(new_alpha, C_num)
        next_den = np.dot(new_alpha, C_den)
        next_lam = next_num / next_den
        
        if np.abs(next_lam - lam) < tol:
            return new_alpha, next_lam
            
        alpha = new_alpha
        lam = next_lam

    return alpha, lam
'''Graphs = ['petster-hamster-household','arenas-email', 'web-polblogs','bio-CE-LC','power-eris1176']
GNAME="petster-hamster-household"
G=BCR.open_graph(GNAME)'''
G=nx.barabasi_albert_graph(100,1) #nx.generators.connected_watts_strogatz_graph(100,49,0.2) 
N= G.number_of_nodes()
adjm=BCR.adjacency(G)
degrees = np.sum(adjm, axis=1)
P = adjm / degrees[:, np.newaxis]
conformist_list=([])
alpha=BCR.conformist_label(conformist_list, N)
eta=BCR.solve_eta_matrix_exactly(adjm)
eta11,eta12,zeta1=BCR.calculate_metrics_with_eta(adjm, alpha, eta)
bcr=eta11/(eta12-zeta1)
print("before: ", bcr)
Ci,Ri=BCR.calculate_CR(P, eta)
conformist_list=[i for i,x in enumerate(Ri) if x<0]
alpha=BCR.conformist_label(conformist_list, N)
eta11,eta12,zeta1=BCR.calculate_metrics_with_eta(adjm, alpha, eta)
bcr=eta11/(eta12-zeta1)
print("after: ", bcr)
#alpha,bcr=solve_minimal_ST(Ci,Ri, 100, max_iter=100, tol=1e-6)
#alphabest,bcrbest=BCR.solve_min_conformists_unconstrained(Ci, Ri, max_iter=100, tol=1e-8)
#number_of_best=(N-sum(alphabest))/N