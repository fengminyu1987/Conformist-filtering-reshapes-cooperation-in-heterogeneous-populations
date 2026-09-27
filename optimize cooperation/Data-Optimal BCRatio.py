# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 07:10:05 2026

@author: ziyan
"""
import BCR
import networkx as nx
import numpy as np
import os
import pandas as pd
# 确保保存目录存在
save_dir = ".\\saves"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
rev = True #True or False: True表示降序，先选取排名高的节点，False表示升序，先选取排名低的节点
selfloop = True #True表示不消除自环，False表示消除自环
Graphs = ['petster-hamster-household','arenas-email', 'web-polblogs','bio-CE-LC','power-eris1176']
GNAME = 'power-eris1176' # 你可以根据需要修改这个变量
G = BCR.open_graph(GNAME,selfloop)
G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute='old_label')
N = G.number_of_nodes()

# 获取邻接矩阵和基本参数
adjm = BCR.adjacency(G)
degrees = np.sum(adjm, axis=1)
# 防止除以0，虽然连通图中度通常不为0
#degrees[degrees == 0] = 1 
P = adjm / degrees[:, np.newaxis]

# 计算核心矩阵和向量
print(f"Calculating Eta and CR vectors for {GNAME}...")
eta = BCR.solve_eta_matrix_exactly(adjm)
Ci, Ri = BCR.calculate_CR(P, eta)

# 定义辅助函数：根据从众者列表生成alpha向量
# alpha=0 代表从众者，alpha=1 代表利益驱动者
def get_alpha_vector(conformist_indices, n_nodes):
    alpha = np.ones(n_nodes)
    if len(conformist_indices) > 0:
        alpha[list(conformist_indices)] = 0
    return alpha

# ==========================================
# 1. 计算各种中心性指标并对节点排序
# ==========================================
print("Calculating Centralities...")

# 1.1 度中心性 (Degree)
# 注意：nx.degree_centrality 返回归一化的度，直接用 G.degree() 也可以，排序结果一样
deg_dict = dict(G.degree())
nodes_sorted_by_degree = sorted(deg_dict, key=deg_dict.get, reverse=rev)

# 1.2 介数中心性 (Betweenness)
print(" - Calculating Betweenness...")
bet_dict = nx.betweenness_centrality(G)
nodes_sorted_by_betweenness = sorted(bet_dict, key=bet_dict.get, reverse=rev)

# 1.3 接近中心性 (Closeness)
print(" - Calculating Closeness...")
clo_dict = nx.closeness_centrality(G)
nodes_sorted_by_closeness = sorted(clo_dict, key=clo_dict.get, reverse=rev)

# 1.4 特征向量中心性 (Eigenvector)
print(" - Calculating Eigenvector...")
try:
    eig_dict = nx.eigenvector_centrality(G, max_iter=1000)
    nodes_sorted_by_eigenvector = sorted(eig_dict, key=eig_dict.get, reverse=rev)
except:
    print("   Eigenvector centrality failed to converge, using Degree instead as fallback.")
    nodes_sorted_by_eigenvector = nodes_sorted_by_degree

# 1.5 PageRank
print(" - Calculating PageRank...")
pr_dict = nx.pagerank(G)
nodes_sorted_by_pagerank = sorted(pr_dict, key=pr_dict.get, reverse=rev)

# 1.6 K Shell
print(" - Calculating K Shell...")
# 返回字典 {node_id: k_shell_value}
coreness = nx.core_number(G) 
# 策略：按 k-shell 值从大到小排序
node_sorted_by_kshell = sorted(G.nodes(), key=lambda n: coreness[n], reverse=rev)
# 将所有策略放入字典方便遍历

# 1.7 Collective Influence
print(" - Calculating Collective Influence...")
def collective_influence(G, l=2):
    ci = {}
    for node in G.nodes():
        # 计算 Ball(node, l) 边界上的节点
        # 这里为了简化，可以用 networkx 的 ego_graph 或者 bfs
        # CI_i = (k_i - 1) * Sum(k_j - 1) for j in boundary
        degree_i = G.degree(node)
        if degree_i <= 1:
            ci[node] = 0
            continue
            
        # 获取 L 阶邻居 (BFS)
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=l)
        boundary_nodes = [n for n, dist in lengths.items() if dist == l]
        
        sum_kj = sum([G.degree(n) - 1 for n in boundary_nodes])
        ci[node] = (degree_i - 1) * sum_kj
    return ci

ci_scores = collective_influence(G, l=2)
node_sorted_by_ci = sorted(G.nodes(), key=lambda n: ci_scores[n], reverse=True)
# 'Optimal' 是你原本的算法，单独处理
strategies = {
    'Degree': nodes_sorted_by_degree,
    'Betweenness': nodes_sorted_by_betweenness,
    'Closeness': nodes_sorted_by_closeness,
    'Eigenvector': nodes_sorted_by_eigenvector,
    'PageRank': nodes_sorted_by_pagerank, 
    'K-Shell': node_sorted_by_kshell,
    'CI': node_sorted_by_ci
}

# ==========================================
# 2. 计算阈值曲线
# ==========================================
xaxis = [i/N for i in range(N)] # 共享 X 轴

data_storage = {
    "fraction_conformists": [i/N for i in range(N)]
}

# --- 2.1 计算理论最优 (Optimal) ---
print("Calculating Optimal...")
y_optimal = []
for i in range(N):
    alphaopt, bcr_val = BCR.solve_minimal_ST(Ci, Ri, i, max_iter=100, tol=1e-6)
    y_optimal.append(bcr_val)

# 存入字典
data_storage["Optimal"] = y_optimal

# --- 2.2 计算中心性策略 ---
for strategy_name, sorted_nodes in strategies.items():
    print(f"Calculating {strategy_name}...")
    y_values = []
    for k in range(N):
        # 你的阈值计算逻辑
        current_conformists = sorted_nodes[:k]
        alpha = np.ones(N)
        if len(current_conformists) > 0:
            alpha[list(current_conformists)] = 0
            
        num = np.dot(alpha, Ci)
        den = np.dot(alpha, Ri)
        
        if den <= 1e-9:
            val = np.nan  # 建议用 NaN 代表无意义，画图时会自动断开，比 1000 更好
        else:
            val = num / den
        y_values.append(val)
    
    # 存入字典
    data_storage[strategy_name] = y_values

# ==========================================
# 3. 整合并保存为单个 CSV 文件
# ==========================================
print(f"Saving combined data for {GNAME}...")

# 将字典转换为 DataFrame
df = pd.DataFrame(data_storage)

# 保存文件
# index=False 表示不保存 0,1,2... 这样的行号
if rev == True:
    if selfloop == False:
        csv_filename = os.path.join(save_dir, f"{GNAME}_bcr.csv")
    elif selfloop == True:
        csv_filename = os.path.join(save_dir, f"{GNAME}_bcr_self.csv")
elif rev == False:
    csv_filename = os.path.join(save_dir, f"{GNAME}_bcr_asc.csv")
df.to_csv(csv_filename, index=False, float_format='%.8f')

print(f"Successfully saved: {csv_filename}")
print(df.head()) # 打印前几行看看样子