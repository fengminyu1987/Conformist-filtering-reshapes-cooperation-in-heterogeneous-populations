# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 23:03:50 2026

@author: ziyan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import string
import json
import os
import BCR  # 确保你的库已导入

def plot_batch_heatmaps(graphs_list,gnames, csv_dir=".\\saves\\optimal_nodes"):
    # 创建 2 行 5 列的画布，constrained_layout 能自动处理子图间距，非常适合学术图表
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    axes = axes.flatten()
    optimal_conformist_ratio = {"petster-hamster-household": 0.7871853546910755, "arenas-email": 0.8658428949691086, "web-polblogs": 0.7465007776049767, "bio-CE-LC": 0.45720040281973817, "power-eris1176": 0.858603066439523, "iceland": 0.45333333333333337, "ca-sandi_auths": 0.7093023255813953, "soc-karate": 0.9411764705882353, "contiguous-usa": 0.9591836734693877, "dolphins": 0.8387096774193549}
    # 设置全局字体和线条样式（提升高级感）
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.linewidth': 1.2
    })

    for i, g_name in enumerate(graphs_list):
        ax = axes[i]
        
        # --- 1. 添加 Nature/PNAS 风格的面板编号 (a, b, c...) ---
        panel_label = string.ascii_lowercase[i]
        g_label = gnames[i]
        # 将字母放在子图的左上角外部一点点，加粗
        ax.text(-0.05, 1.05, f"{panel_label}", transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='bottom', ha='right')
        ax.set_title(g_label, fontsize=14, pad=10)
        # --- 2. 加载网络图和 CSV 数据 ---
        csv_path = os.path.join(csv_dir, f"{g_name}.csv")
        
        try:
            # 打开图 G
            G = BCR.open_graph(g_name)
            N = G.number_of_nodes()
            
            # 读取数据
            df = pd.read_csv(csv_path)
            df['optimal_node_set'] = df['optimal_node_set'].apply(ast.literal_eval)
        except Exception as e:
            ax.text(0.5, 0.5, f"Data Error\n{g_name}", ha='center', va='center')
            ax.axis('off')
            continue
        # --- 3. 处理节点排序和矩阵生成 ---
        # 计算度并排序：最左侧为最大 Hub
        node_degrees = sorted(dict(G.degree()).items(), key=lambda item: item[1], reverse=True)
        rank_map = {node_id: rank for rank, (node_id, degree) in enumerate(node_degrees)}
        
        # 初始化 N*N 矩阵
        matrix = np.zeros((len(df), N))
        
        # 填充矩阵
        for row_idx, row in df.iterrows():
            conformists = row['optimal_node_set']
            # 过滤掉可能的异常值
            if isinstance(conformists, list):
                ranked_indices = [rank_map[node] for node in conformists if node in rank_map]
                matrix[row_idx, ranked_indices] = 1

        # --- 4. 绘制热力图 ---
        # 蓝色调：白色为利益驱动(0)，深蓝为从众者(1)
        ax.imshow(matrix, aspect='auto', cmap='Blues', origin='lower', extent=[0, N, 0, 1])
        ax.axhline(optimal_conformist_ratio[g_name], color='red',ls='--',lw=1.5,alpha=0.6, zorder=3)
        # --- 5. 坐标轴与边框美化 (关键学术风格控制) ---
        # 隐藏上方和右方的边框线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 只在第一列显示 Y 轴标签
        if i % 5 == 0:
            ax.set_ylabel(r"Conformist Ratio Constraint", fontsize=14)
        else:
            # 隐藏内部图的 Y 轴刻度标签，保持画面清爽
            ax.tick_params(labelleft=False) 
            
        # 只在最后一行显示 X 轴标签
        if i >= 5:
            ax.set_xlabel("Nodes (Sorted by Degree)", fontsize=13)
        ax.tick_params(labelbottom=True)

    # 保存高斯真图 (pdf 矢量图最适合插入论文)
    save_path = "SI_Optimal_Heatmaps_Panel.pdf"
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    print(f"绘图完成！图片已保存至: {save_path}")
    plt.show()

# --- 运行代码 ---
Graphs = [
    'petster-hamster-household', 'arenas-email', 'web-polblogs', 
    'bio-CE-LC', 'power-eris1176', 'iceland', 
    'ca-sandi_auths', 'soc-karate', 'contiguous-usa', 'dolphins'
]
Gnames = ["Hamster Owners", "Email", "PolBlogs", "WormNet", "Power Grid",
          'Iceland','Sandia National Labs', 'Social Karate Club','USA Contiguous Border', 'Dolphins']
# 调用函数 (假设你的 CSV 保存在上一段代码默认的 optimal_nodes_results 文件夹中)
plot_batch_heatmaps(Graphs,Gnames, csv_dir=".\\saves\\optimal nodes")