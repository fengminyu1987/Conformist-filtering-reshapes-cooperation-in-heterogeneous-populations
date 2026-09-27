# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 22:31:16 2026

@author: ziyan
"""

import numpy as np
import pandas as pd
import networkx as nx
import os
from tqdm import tqdm
import json

# 确保 BCR 库已经被正确导入
import BCR 

def get_bcr_matrices(G):
    """
    根据给定的图 G 计算 C_num 和 C_den
    """
    adjm = BCR.adjacency(G)
    degrees = np.sum(adjm, axis=1)
    P = adjm / degrees[:, np.newaxis]
    eta = BCR.solve_eta_matrix_exactly(adjm)
    Ci, Ri = BCR.calculate_CR(P, eta)
    return Ci, Ri

def extract_optimal_nodes_for_networks(networks_dict, save_dir=".\\saves\\optimal nodes\\"):
    """
    遍历真实网络，提取每个从众者个数下的最优节点集合并保存为 CSV。
    
    参数:
    networks_dict: 字典，键为网络名称 (字符串)，值为 networkx 图对象。
                   例如: {'Email': G_email, 'PolBlogs': G_polblogs}
    save_dir: 存放 CSV 文件的目标文件夹路径
    """
    # 如果保存目录不存在，则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for net_name, G in networks_dict.items():
        N = G.number_of_nodes()
        print(f"开始处理网络: {net_name} (规模 N={N})")
        
        # 1. 一次性计算该网络的 C_num 和 C_den，避免在循环中重复计算
        print("  -> 正在计算 BCR 矩阵...")
        C_num, C_den = get_bcr_matrices(G)
        
        results = []
        
        # 2. 遍历从众者个数 k 从 0 到 N-1
        # 使用 tqdm 包装 range(N) 以显示进度条
        for k in tqdm(range(N), desc=f"  -> 优化进度 ({net_name})"):
            try:
                # 调用 BCR 库求解当前 k 个从众者下的最优配置
                alpha, lam = BCR.solve_minimal_ST(C_num, C_den, k, max_iter=100, tol=1e-6)
                
                # 提取最优节点集：在 alpha 中值为 0 的索引
                # 假设 alpha 是一维 numpy 数组
                conformist_indices = np.where(np.array(alpha) == 0)[0].tolist()
                
                # 计算从众者比例
                rho = k / N
                
                # 将结果添加到列表
                # 注意：将列表转换为字符串存储，避免 pandas 保存时出现格式混乱
                results.append({
                    'conformist_ratio': rho,
                    'optimal_node_set': str(conformist_indices)
                })
                
            except Exception as e:
                # 防止由于某个特定的 k 不收敛导致整个网络的数据丢失
                print(f"\n警告: 在网络 {net_name} 中，k={k} 时发生错误: {e}")
                results.append({
                    'conformist_ratio': k / N,
                    'optimal_node_set': "ERROR"
                })
                
        # 3. 将结果转化为 DataFrame 并保存为单独的 CSV 文件
        df = pd.DataFrame(results)
        save_path = os.path.join(save_dir, f"{net_name}.csv")
        df.to_csv(save_path, index=False)
        print(f"完成！数据已保存至: {save_path}\n" + "-"*50)


# ==========================================
# 示例运行模块 (你可以根据实际情况替换这部分)
# ==========================================
if __name__ == "__main__":
    #Graphs = ['petster-hamster-household','arenas-email', 'web-polblogs','bio-CE-LC','power-eris1176'] #big sizes
    Graphs = ['petster-hamster-household','arenas-email', 'web-polblogs','bio-CE-LC','power-eris1176','iceland','ca-sandi_auths','soc-karate','contiguous-usa', 'dolphins'] #small sizes
    nets = {}
    for i in Graphs:
        G = BCR.open_graph(i)
        nets[i] = G
    # 执行主程序
    #extract_optimal_nodes_for_networks(nets)
    best_conformist_ratio={}
    for i in Graphs:
        g = nets[i]
        Ci,Ri = get_bcr_matrices(g)
        alpha,lam = BCR.solve_min_conformists_unconstrained(Ci, Ri, max_iter=100, tol=1e-8, epsilon=1e-11)
        best_conformist_ratio[i] = 1-sum(alpha)/g.number_of_nodes()
    json_str = json.dumps(best_conformist_ratio)
    filename = open(".\\saves\\optimal nodes\\optimal_conformist_ratio.json", "w")
    filename.write(json_str)