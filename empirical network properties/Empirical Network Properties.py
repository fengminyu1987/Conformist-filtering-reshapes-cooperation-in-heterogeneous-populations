# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 22:45:30 2026

@author: ziyan
"""

import numpy as np
import pandas as pd
import networkx as nx
import os
import BCR
# 假设你的 BCR 模块已经正确导入
# import BCR 

def get_bcr_matrices(G):
    """
    根据给定的图 G 计算 C_num 和 C_den
    """
    adjm = BCR.adjacency(G)
    degrees = np.sum(adjm, axis=1)
    # 避免除以0的警告，将度数为0的节点的度数设为1（或者根据你的网络实际情况处理孤立点）
    degrees_safe = np.where(degrees == 0, 1, degrees)
    P = adjm / degrees_safe[:, np.newaxis]
    eta = BCR.solve_eta_matrix_exactly(adjm)
    Ci, Ri = BCR.calculate_CR(P, eta)
    return Ci, Ri

# 文件名和图名列表
Graphs = ['petster-hamster-household', 'arenas-email', 'web-polblogs', 'bio-CE-LC', 
          'power-eris1176', 'iceland', 'ca-sandi_auths', 'soc-karate', 
          'contiguous-usa', 'dolphins']

Gnames = ["Hamster Owners", "Email", "PolBlogs", "WormNet", "Power Grid",
          "Iceland", "Sandia National Labs", "Social Karate Club", 
          "USA Contiguous Border", "Dolphins"]

results = []
dataset_path = ""

print("开始处理网络数据并计算拓扑特征...")

for file_name, net_name in zip(Graphs, Gnames):
    file_path = os.path.join(dataset_path, f"{file_name}") # 假设是txt格式
    
    try:
        # 1. 打开图
        G_bcr = BCR.open_graph(file_path, selfloop=False)
        
        # 2. 计算合作阈值 (b/c)*
        Ci, Ri = get_bcr_matrices(G_bcr)
        threshold = np.sum(Ci) / np.sum(Ri)
        
        # 3. 提取邻接矩阵并转换为 NetworkX 图以计算标准拓扑特征
        adjm = BCR.adjacency(G_bcr)
        G_nx = nx.from_numpy_array(adjm)
        
        # 如果图是有向的，根据你的博弈模型决定是否转为无向图计算拓扑
        # 这里默认当作无向图处理
        
        # --- 计算拓扑指标 ---
        N = G_nx.number_of_nodes()
        E = G_nx.number_of_edges()
        
        degrees = [d for n, d in G_nx.degree()]
        k_avg = np.mean(degrees)
        k_var = np.var(degrees)
        # 异质性参数 <k^2> / <k>
        heterogeneity = (k_var + k_avg**2) / k_avg if k_avg > 0 else 0
        
        # 平均聚类系数 C
        clustering_coeff = nx.average_clustering(G_nx)
        
        # 同配系数 r
        assortativity = nx.degree_assortativity_coefficient(G_nx)
        
        # 最大连通子图和平均最短路径长度 L
        # 注意：如果网络不连通，计算全局最短路径会报错，因此我们在最大连通子图上计算
        if len(G_nx) > 0:
            largest_cc = max(nx.connected_components(G_nx), key=len)
            G_lcc = G_nx.subgraph(largest_cc)
            # L 计算可能比较耗时，特别是对于 Email 或 Hamster 网络
            L = nx.average_shortest_path_length(G_lcc)
        else:
            L = np.nan
        # Global Responsiveness
        Ci,Ri = get_bcr_matrices(G_nx)
        R = sum(Ri)
        # 记录结果
        results.append({
            "Network": net_name,
            "$N$": N,
            "$E$": E,
            "$\\langle k \\rangle$": round(k_avg, 2),
            "$\\langle k^2 \\rangle / \\langle k \\rangle$": round(heterogeneity, 2),
            "$C$": round(clustering_coeff, 4),
            "$r$": round(assortativity, 4),
            "$L$": round(L, 2),
            "$(b/c)^*$": round(threshold, 4), 
            "$R$": round(R)
        })
        print(f"[{net_name}] 处理完成。")
        
    except Exception as e:
        print(f"处理网络 {net_name} 时出错: {e}")

# 4. 生成数据框并输出 LaTeX 表格
df = pd.DataFrame(results)

print("\n\n--- 适用于 SI 的 LaTeX 表格代码 ---")
# escape=False 保证 LaTeX 公式中的 $ 和 \ 不会被转义
print(df.to_latex(index=False, escape=False, float_format="%.4f", column_format="lrrrrrrrr"))

# 如果你想同时保存一份 CSV 到本地
df.to_csv("network_properties.csv", index=False)