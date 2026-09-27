# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 00:11:10 2026

@author: ziyan
"""

import numpy as np
import networkx as nx
import time
import json
from scipy import sparse
from numba import njit
from joblib import Parallel, delayed
import BCRatio
# --- 1. 核心 Numba 模拟函数 ---
@njit(fastmath=True)
def get_fitness_numba(node_idx, node_state, b, c, delta, 
                      nl_data, nl_indices, nl_indptr):
    """
    计算单个节点的适应度 (Fitness)
    公式: exp(delta * (-c * (1-S_i) + b * sum(W_ij * (1-S_j))))
    """
    # 1. 自身的贡献 (-c 如果是背叛者 D(1), 0 如果是合作者 C(0))
    # 注意：原代码逻辑 node_state[node]==0 是合作者(C)
    # 收益：-c * (1 - node_state[node]) 意味着如果是 0 (C)，这一项是 -c；如果是 1 (D)，这一项是 0。
    # 等等，原代码 result=-c*(1-node_state[node])。
    # 如果 state=0(C), result = -c * (1-0) = -c. (合作者付出成本)
    # 如果 state=1(D), result = -c * (1-1) = 0.
    
    payoff_cost = -c * (1.0 - node_state[node_idx])
    
    # 2. 从邻居获得的收益 (b * sum(W_ij * (1-S_j)))
    start = nl_indptr[node_idx]
    end = nl_indptr[node_idx + 1]
    
    benefit_sum = 0.0
    
    # 遍历邻居 (利用 CSR 结构)
    for k in range(start, end):
        neighbor = nl_indices[k]
        weight = nl_data[k]
        
        # 邻居是合作者 (0) 才能提供收益
        if node_state[neighbor] == 0:
            benefit_sum += weight * 1.0 # (1 - 0)
            
    total_payoff = payoff_cost + b * benefit_sum
    return np.exp(delta * total_payoff)

@njit(fastmath=True)
def run_simulation_kernel(N, b, c, delta, 
                          adj_indices, adj_indptr,           # 图结构
                          nl_data, nl_indices, nl_indptr,    # 归一化拉普拉斯权重
                          conformist_arr,                    # bool 数组
                          max_steps=100000000):
    """
    单次模拟内核
    返回: 1 (C 固定), 0 (D 固定)
    """
    # 初始化状态: 全 D (1)
    node_state = np.ones(N, dtype=np.int8)
    
    # 随机选择一个初始突变点变为 C (0)
    init_node = np.random.randint(0, N)
    node_state[init_node] = 0
    
    # 追踪合作者数量，避免反复 mean()
    coop_count = 1
    
    steps = 0
    while 0 < coop_count < N:
        steps += 1
        if steps > max_steps: 
            return 0 # 防止死循环，视作失败
            
        # 1. 选择更新者 (Updater)
        updater = np.random.randint(0, N)
        old_state = node_state[updater]
        new_state = old_state # 默认不变
        
        # 2. 判断更新逻辑
        is_conformist = conformist_arr[updater]
        
        # --- 从众者 (Conformist) 逻辑 ---
        if is_conformist:
            # 随机选择一个邻居模仿
            u_start = adj_indptr[updater]
            u_end = adj_indptr[updater + 1]
            if u_end > u_start:
                rand_idx = np.random.randint(u_start, u_end)
                neighbor = adj_indices[rand_idx]
                new_state = node_state[neighbor]
                
        # --- 利益驱动者 (Payoff-driven) 逻辑 ---
        else:
            # 计算邻居的总适应度
            u_start = adj_indptr[updater]
            u_end = adj_indptr[updater + 1]
            
            k_cfc = 0.0 # C 邻居的适应度之和
            k_dfd = 0.0 # D 邻居的适应度之和
            
            for k in range(u_start, u_end):
                neighbor = adj_indices[k]
                
                # 计算该邻居的 Fitness
                fit = get_fitness_numba(neighbor, node_state, b, c, delta, 
                                      nl_data, nl_indices, nl_indptr)
                
                if node_state[neighbor] == 0: # Neighbor is C
                    k_cfc += fit
                else: # Neighbor is D
                    k_dfd += fit
            
            total_fit = k_cfc + k_dfd
            if total_fit > 1e-12:
                prob_switch_to_c = k_cfc / total_fit
                if np.random.random() < prob_switch_to_c:
                    new_state = 0
                else:
                    new_state = 1
        
        # 3. 更新状态计数
        if new_state != old_state:
            node_state[updater] = new_state
            if new_state == 0:
                coop_count += 1
            else:
                coop_count -= 1
                
    # 循环结束
    return 1 if coop_count == N else 0

@njit(parallel=True)
def run_batch_simulation(n_sims, N, b, c, delta, 
                         adj_indices, adj_indptr, 
                         nl_data, nl_indices, nl_indptr, 
                         conformist_arr):
    """
    Numba 内部也可以并行，但为了更好的随机数控制和配合 Joblib，
    这里通常作为串行 Batch 运行，由 Joblib 在外部并行。
    """
    fixation_count = 0
    for _ in range(n_sims):
        res = run_simulation_kernel(N, b, c, delta, 
                                    adj_indices, adj_indptr, 
                                    nl_data, nl_indices, nl_indptr, 
                                    conformist_arr)
        fixation_count += res
    return fixation_count

# --- 2. 辅助数据处理函数 ---

def prepare_graph_data(G, conformist_list):
    """
    将 NetworkX 图和相关数据转换为 CSR 数组供 Numba 使用
    """
    N = G.number_of_nodes()
    
    # 1. 邻接矩阵 CSR (用于寻找邻居)
    adj_mat = nx.to_scipy_sparse_array(G, format='csr')
    
    # 2. 归一化拉普拉斯矩阵 (用于计算 Payoff)
    # 按照你原代码的逻辑：D^-1 * A
    degrees = np.array(adj_mat.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0 # 避免除零
    inv_degrees = 1.0 / degrees
    D_inv = sparse.diags(inv_degrees)
    nl_mat = D_inv @ adj_mat # 稀疏矩阵乘法
    nl_mat = nl_mat.tocsr()  # 确保是 CSR 格式
    
    # 3. Conformist 掩码数组
    # 原代码 conformist_label 返回 0 代表是 conformist? 
    # 原代码: result=np.ones(N); for i in list: result[i]=0
    # 但在 DB_update 里: if changer in conformist_list (即 list 里的才是 conformist)
    # 所以我们要生成一个 bool 数组，True 代表是 Conformist
    conformist_mask = np.zeros(N, dtype=bool)
    for node in conformist_list:
        if node < N:
            conformist_mask[node] = True
            
    return {
        'N': N,
        'adj_indices': adj_mat.indices,
        'adj_indptr': adj_mat.indptr,
        'nl_data': nl_mat.data,
        'nl_indices': nl_mat.indices,
        'nl_indptr': nl_mat.indptr,
        'conformist_mask': conformist_mask
    }

# --- 3. 主程序逻辑 ---

def run_experiment_fast(G, conformist_list,delta, b_values, avgs=5000000, n_jobs=10):
    c = 1.0
    
    # 1. 数据预处理
    print("Pre-processing graph data...")
    data = prepare_graph_data(G, conformist_list)
    
    results = {}
    
    # 2. 遍历参数 b
    for b in b_values:
        start_time = time.time()
        
        # 将总任务数 avgs 分配给 n_jobs
        batch_size = avgs // n_jobs
        remainder = avgs % n_jobs
        tasks = [batch_size] * n_jobs
        if remainder > 0:
            tasks[0] += remainder
            
        print(f"Simulating b={b:.5f} with {avgs} runs...", end=" ")
        
        # 3. 并行执行 Batch
        # delayed 函数接收 Numba 需要的所有原生数组
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(run_batch_simulation)(
                n_sims, 
                data['N'], 
                float(b), c, delta,
                data['adj_indices'], data['adj_indptr'],
                data['nl_data'], data['nl_indices'], data['nl_indptr'],
                data['conformist_mask']
            ) for n_sims in tasks
        )
        
        total_fixations = sum(batch_results)
        ratio = total_fixations / avgs
        results['{:.5f}'.format(b)] = ratio
        
        elapsed = time.time() - start_time
        print(f"Done. Ratio: {ratio:.5f}, Time: {elapsed:.2f}s")
        
    return results
def conformist_on_degree(G,a,type_):
    n = G.number_of_nodes()
    N = int(np.floor(n * a))
    degrees = [(node, G.degree(node)) for node in G.nodes()]
    if type_=='large':
        degrees.sort(key=lambda x: (-x[1], x[0]))
        return [node for node, _ in degrees[:N]]
    if type_=='small':
        degrees.sort(key=lambda x: (x[1], x[0]))
        return [node for node, _ in degrees[:N]]
def open_graph(network_type):
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
    return G
# --- 4. 示例调用 (模拟你的环境) ---
if __name__ == "__main__":
    Gs=["iceland", "ca-sandi_auths", "soc-karate", "contiguous-usa"]
    net="ca-sandi_auths"
    conformist_type='small'#'small'
    conformist_ratio = 0.1 #0
    delta = 10  
    njbs = 5
    print(conformist_type, conformist_ratio, delta)
    print("Generating Graph...",net)
    G = open_graph(net)
    N=G.number_of_nodes()
    adjm=BCRatio.adjacency(G)
    degrees = np.sum(adjm, axis=1)
    # 模拟 conformist
    conformist_list = conformist_on_degree(G, conformist_ratio, conformist_type)
    alpha=BCRatio.conformist_label(conformist_list,N)
    # 设置 b 的范围
    # 假设 theorem 算出来是 1.2
    #theorem = BCRatio.Get_BCR(G, alpha)
    #B = list(np.linspace(theorem-2, theorem+2, 9))  
    B = list(np.linspace(1, 11, 20))
    # 运行 
    final_results = run_experiment_fast(G, conformist_list,delta, B, avgs=5000000, n_jobs=njbs) # 测试用 avgs 设小点
    print("\n saving...")
    json_str=json.dumps(final_results)
    with open('.\\saves\\{}_{}_{:.2f}_{}_{:.2f}.json'.format(net,N,conformist_ratio,conformist_type, delta), 'w') as json_file:
        json_file.write(json_str)