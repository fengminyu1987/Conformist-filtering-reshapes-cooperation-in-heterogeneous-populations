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

def Get_BC_Ratio(adj_matrix, alpha):
    eta = solve_eta_matrix_exactly(adj_matrix)
    eta11, eta12, zeta1 = calculate_metrics_with_eta(adj_matrix, alpha, eta)
    return (eta11)/(eta12-zeta1)


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
def open_graph(network_type):
    filename='.\\saves\\datasets\\'+network_type+'.txt'
    G=nx.Graph()
    with open(filename) as file:
        for line in file:
            head, tail=[str(x) for x in line.split()]
            G.add_edge(int(head),int(tail))
    return G
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