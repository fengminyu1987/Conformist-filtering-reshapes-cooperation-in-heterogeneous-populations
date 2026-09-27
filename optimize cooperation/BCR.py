# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 06:27:36 2026

@author: ziyan
"""

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import networkx as nx
import time
import random
def adjacency(G):
    nodes = sorted(G.nodes())
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
    return adj_matrix
def solve_eta_matrix_exactly(adj_matrix):
    """
    直接求解 eta 矩阵方程：2*eta - P*eta - eta*P.T + 2*diag(P*eta) = J - I
    这是对递归定义最忠实的直接数值解法。
    """
    n = adj_matrix.shape[0]
    degrees = np.sum(adj_matrix, axis=1)
    P = adj_matrix / degrees[:, np.newaxis]
    
    # 定义线性算子 L(H)
    def matvec(h_flat):
        H = h_flat.reshape((n, n))
        PH = P @ H
        # 核心算子方程
        # 2*H - P*H - H*P.T 保证了 off-diagonal 的递归关系
        # 2*diag(diag(PH)) 是为了强制 H_ii = 0
        res = 2 * H - PH - H @ P.T + 2 * np.diag(np.diag(PH))
        return res.flatten()

    # 构造算子
    A_op = LinearOperator((n*n, n*n), matvec=matvec)
    
    # 右侧项: J - I (即所有 off-diagonal 为 1, 对角线为 0)
    rhs = (np.ones((n, n)) - np.eye(n)).flatten()
    
    # 使用 gmres 求解，这比迭代法更稳健且收敛更快
    # tol 设置得尽可能小以保证精确
    h_sol_flat, exit_code = gmres(A_op, rhs, rtol=1e-12)
    
    if exit_code != 0:
        print("Warning: Solver did not converge perfectly.")
        
    return h_sol_flat.reshape((n, n))

def calculate_metrics_with_eta(adj_matrix, alpha, eta):
    """
    使用解出的 eta 矩阵计算最终指标
    """
    n = adj_matrix.shape[0]
    degrees = np.sum(adj_matrix, axis=1)
    m = np.sum(degrees) / 2
    pi = degrees / (2 * m)
    P = adj_matrix / degrees[:, np.newaxis]
    W = np.diag(alpha)
    
    # 辅助函数：计算 Tr(diag(pi) * M * eta)
    def sum_metric(M):
        # 这里的计算等价于 sum_{j,k} pi_j * M_jk * eta_jk
        return np.sum((pi[:, np.newaxis] * M) * eta)

    # 计算目标指标
    M11 = P @ W @ P
    M12 = P @ W @ (P @ P)
    p_tilde = P @ alpha
    M_zeta = np.diag(p_tilde) @ P
    return sum_metric(M11), sum_metric(M12), sum_metric(M_zeta)
def calculate_gamma(P, eta, b, c):
    """
    计算任意结构图下的每个节点的 gamma_i。
    
    参数:
    P   : np.ndarray, 1步转移概率矩阵 (N x N)
    eta : np.ndarray, 合并时间矩阵 (N x N)
    b   : float, 收益 (benefit)
    c   : float, 成本 (cost)
    
    返回:
    gamma : np.ndarray, 每个节点的强度贡献 (长度为 N 的向量)
    """
    N = P.shape[0]
    
    # 1. 计算平稳分布 pi (基于 P 的特征向量，或对于无向图直接由度决定)
    # 这里使用通用方法：P^T * pi = pi
    evals, evecs = np.linalg.eig(P.T)
    pi = np.real(evecs[:, np.isclose(evals, 1)])
    pi = pi[:, 0] / np.sum(pi) # 归一化
    
    # 2. 计算 P 的 2 步转移矩阵 P2
    P2 = np.matmul(P, P)
    
    # 3. 计算公式中的第一项 (对应 -c 的系数): \sum_{j,k} p_{ij} p_{ik} \eta_{jk}
    # 这可以写成矩阵形式: (P * eta * P^T) 的对角线元素
    # 因为 \sum_j p_{ij} (\sum_k \eta_{jk} p_{ik}) = (P * (eta * P^T))_{ii}
    term_c = np.diag(np.matmul(P, np.matmul(eta, P.T)))
    
    # 4. 计算公式中的第二项 (对应 b 的前半部分): \sum_{j,k} p_{ij}^{(1)} p_{ik}^{(2)} \eta_{jk}
    # 矩阵形式: (P * eta * (P2)^T) 的对角线元素
    term_b1 = np.diag(np.matmul(P, np.matmul(eta, P2.T)))
    
    # 5. 计算公式中的第三项 (对应 b 的后半部分): \sum_{j,k} p_{jk}^{(1)} p_{ij}^{(1)} \eta_{jk}
    # 注意这里指标是 jk, ij。我们可以改写为 \sum_j p_{ij} (\sum_k p_{jk} \eta_{jk})
    # 令 vector_v_i = \sum_k p_{jk} \eta_{jk}，这是矩阵 (P * eta) 的对角线
    # 然后再与 P 的行相乘
    P_eta = np.matmul(P, eta)
    diag_P_eta = np.diag(P_eta) # 这是一个长度为 N 的向量，第 j 个元素是 \sum_k p_{jk} \eta_{jk}
    term_b2 = np.matmul(P, diag_P_eta) # 每个 i 对应 \sum_j p_{ij} * diag_P_eta[j]
    
    # 6. 汇总计算 gamma_i
    gamma =  pi*(-c * term_c + b * (term_b1 - term_b2))
    responsiveness = pi * (term_b1 - term_b2)
    return gamma, responsiveness
def calculate_CR(P, eta):
    N = P.shape[0]
    
    # 1. 计算平稳分布 pi (基于 P 的特征向量，或对于无向图直接由度决定)
    # 这里使用通用方法：P^T * pi = pi
    evals, evecs = np.linalg.eig(P.T)
    pi = np.real(evecs[:, np.isclose(evals, 1)])
    pi = pi[:, 0] / np.sum(pi) # 归一化
    
    # 2. 计算 P 的 2 步转移矩阵 P2
    P2 = np.matmul(P, P)
    
    # 3. 计算公式中的第一项 (对应 -c 的系数): \sum_{j,k} p_{ij} p_{ik} \eta_{jk}
    # 这可以写成矩阵形式: (P * eta * P^T) 的对角线元素
    # 因为 \sum_j p_{ij} (\sum_k \eta_{jk} p_{ik}) = (P * (eta * P^T))_{ii}
    term_c = np.diag(np.matmul(P, np.matmul(eta, P.T)))
    
    # 4. 计算公式中的第二项 (对应 b 的前半部分): \sum_{j,k} p_{ij}^{(1)} p_{ik}^{(2)} \eta_{jk}
    # 矩阵形式: (P * eta * (P2)^T) 的对角线元素
    term_b1 = np.diag(np.matmul(P, np.matmul(eta, P2.T)))
    
    # 5. 计算公式中的第三项 (对应 b 的后半部分): \sum_{j,k} p_{jk}^{(1)} p_{ij}^{(1)} \eta_{jk}
    # 注意这里指标是 jk, ij。我们可以改写为 \sum_j p_{ij} (\sum_k p_{jk} \eta_{jk})
    # 令 vector_v_i = \sum_k p_{jk} \eta_{jk}，这是矩阵 (P * eta) 的对角线
    # 然后再与 P 的行相乘
    P_eta = np.matmul(P, eta)
    diag_P_eta = np.diag(P_eta) # 这是一个长度为 N 的向量，第 j 个元素是 \sum_k p_{jk} \eta_{jk}
    term_b2 = np.matmul(P, diag_P_eta) # 每个 i 对应 \sum_j p_{ij} * diag_P_eta[j]
    return pi*term_c, pi*(term_b1 - term_b2)
def open_graph(network_type, selfloop = True):
    filename='.\\saves\\datasets\\'+network_type+'.txt'
    G=nx.Graph()
    with open(filename) as file:
        for line in file:
            head, tail=[str(x) for x in line.split()]
            G.add_edge(int(head),int(tail))
    if 0 not in G.nodes():#relabel to unify node labels starting from 0
        j = 0
        new_mapping = {}
        for i in G.nodes:
            new_mapping[i] = j
            j += 1
        G = nx.relabel_nodes(G, new_mapping)
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    g = G.subgraph(largest_cc_nodes).copy()
    if selfloop == False:
        g.remove_edges_from(nx.selfloop_edges(g))
    return g
def conformist_label(conformist_list,N):
    result=np.ones(N)
    for i in conformist_list:
        result[i]=0
    return result
def Generate_Star_HH(N):
    g1=nx.star_graph(N-1)
    g2=nx.star_graph(N-1)
    G=nx.disjoint_union(g1,g2)
    G.add_edge(0,N)
    return G
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
def solve_mininal_unconstrained(C_num, C_den, max_iter=100, tol=1e-8):
    """
    使用 Dinkelbach 算法寻找全局最优的从众者配置（不限制数量）。
    目标：最小化 (b/c)* = (alpha @ C_num) / (alpha @ C_den)
    
    参数:
    C_num: 分子系数向量 (Cost term coefficients)
    C_den: 分母系数向量 (Benefit term coefficients)
    """
    n = len(C_num)
    
    # --- 1. 初始化 ---
    # 策略：先全选，或者随机选几个，只要分母不为0即可
    # 这里我们采用一种启发式：选所有 Benefit (C_den) 为正的点作为初始集
    alpha = (C_den > 0).astype(float)
    
    # 防止初始 alpha 全为 0 的极端情况
    if np.sum(alpha) == 0:
        # 如果没有正贡献的点，随便选一个 C_num 最小的点启动
        alpha[np.argmin(C_num)] = 1.0

    # 计算初始 lambda
    curr_num = np.dot(alpha, C_num)
    curr_den = np.dot(alpha, C_den)
    
    if curr_den <= 0:
        # 如果这一步发生，说明整个网络结构极难合作，
        # 或者你需要检查输入数据是否正确
        return None, np.inf
        
    lam = curr_num / curr_den
    
    for iteration in range(max_iter):
        # --- 2. 构造权重 w ---
        # w_i = C_num_i - lambda * C_den_i
        w = C_num - lam * C_den
        
        # --- 3. 求解子问题 (无约束选择) ---
        # 逻辑：只要 w_i < 0 就选，w_i >= 0 就不选
        new_alpha = (w < 0).astype(float)
        
        # --- 4. 防退化处理 (关键) ---
        # 如果 w 全部大于 0，算法会倾向于选择“空集”(全0)，
        # 但在物理上我们需要至少一个从众者来定义阈值。
        # 此时我们强制选择 w 最小的那个节点（最接近负值的点）。
        if np.sum(new_alpha) == 0:
            best_idx = np.argmin(w)
            new_alpha[best_idx] = 1.0
            
        # --- 5. 计算当前目标函数值 ---
        # F(lambda) = sum(alpha * w)
        # 实际上 Dinkelbach 的收敛标准通常看 lambda 的变化，或者 F 是否接近 0
        current_obj_val = np.dot(new_alpha, w)
        
        # --- 6. 更新 lambda ---
        next_num = np.dot(new_alpha, C_num)
        next_den = np.dot(new_alpha, C_den)
        
        # 安全检查：分母必须为正
        if next_den <= 1e-10:
            print(f"Warning: Denominator vanished or negative at iter {iteration}. Returning best previous.")
            return alpha, lam
            
        next_lam = next_num / next_den
        
        # --- 7. 检查收敛 ---
        # 如果 lambda 不再变化，或者 F(lambda) 极小
        if np.abs(next_lam - lam) < tol and np.abs(current_obj_val) < tol:
            # print(f"Converged at iteration {iteration}")
            return new_alpha, next_lam
            
        # 更新状态
        alpha = new_alpha
        lam = next_lam

    return alpha, lam
def solve_min_conformists_unconstrained(C_num, C_den, max_iter=100, tol=1e-8, epsilon=1e-11):
    """
    在保证 (b/c)* 最小的前提下，最小化从众者数量（即最大化 alpha=1 的数量）。
    """
    n = len(C_num)
    # 初始化：假设全是利益驱动者
    alpha = np.ones(n)
    
    curr_num = np.dot(alpha, C_num)
    curr_den = np.dot(alpha, C_den)
    lam = curr_num / curr_den
    
    for iteration in range(max_iter):
        w = C_num - lam * C_den
        
        # --- 核心修正处 ---
        # 逻辑：只要这个节点不拖后腿 (w <= 0)，或者拖后腿的程度在容忍范围内 (w < epsilon)
        # 我们就让他保持为“利益驱动者” (alpha = 1)
        new_alpha = (w < epsilon).astype(float) 
        
        # 防退化：如果全变成了从众者(alpha全0)，强制保留贡献最大的那个为利益驱动者
        if np.sum(new_alpha) == 0:
            new_alpha[np.argmin(w)] = 1.0
            
        next_num = np.dot(new_alpha, C_num)
        next_den = np.dot(new_alpha, C_den)
        next_lam = next_num / next_den
        
        if np.abs(next_lam - lam) < tol:
            return new_alpha, next_lam
            
        alpha = new_alpha
        lam = next_lam

    return alpha, lam