# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 17:53:27 2026

@author: ziyan
"""
import matplotlib.pyplot as plt
import os
import pandas as pd
def plot_si_scatter_panels(ratios=[0.1, 0.2], networks=['BA', 'WS', 'HK']):
    heuristics = ['Degree', 'CI', 'K-Shell', 'Betweenness', 'Closeness', 'Eigenvector', 'PageRank']
    
    # 创建 2x3 的画布
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    
    # 定义“英雄与背景”的颜色
    color_bg = '#B0BEC5'  # 高级灰
    color_hero = '#D32F2F' # 沉稳的亮红
    
    for row_idx, ratio in enumerate(ratios):
        for col_idx, net in enumerate(networks):
            ax = axes[row_idx, col_idx]
            file_path = f"./saves/{net}_{ratio}_Centrality_Compare.csv"
            
            if not os.path.exists(file_path):
                ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center')
                continue
                
            df = pd.read_csv(file_path)
            x_val = df['initial_bcr']
            
            # 1. 画背景云 (所有启发式指标)
            for h in heuristics:
                ax.scatter(x_val, df[h], color=color_bg, alpha=0.15, s=3, edgecolors='none', zorder=1)
                
            # 2. 画英雄点 (Optimal)
            ax.scatter(x_val, df['optimal_bcr'], color=color_hero, alpha=0.8, s=8, edgecolors='none', zorder=2)
            
            # 3. 画 Y=X 参考线 (无干预基准线)
            min_val = min(x_val.min(), df['optimal_bcr'].min())
            max_val = x_val.max()
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.7, zorder=0)
            
            # 4. 设置标签与标题
            if row_idx == 0:
                ax.set_title(f"{net} Network", fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"$(b/c)^*$ ($\\rho={ratio}$)")
            if row_idx == 1:
                ax.set_xlabel(r"Initial Threshold $(b/c)_{initial}$")
                
            # 去除冗余边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # 5. 添加全局唯一图例
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_bg, markersize=8, alpha=0.6),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_hero, markersize=8),
        Line2D([0], [0], color='k', linestyle='--', linewidth=1.5)
    ]
    fig.legend(custom_lines, ['Centrality Heuristics', 'Optimal Formulation', 'Baseline (No intervention)'], 
               loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)

    plt.tight_layout()
    plt.savefig('SI_Optimality_Scatter.pdf', format='pdf', bbox_inches='tight',dpi=500)
    plt.show()

# 执行画图
plot_si_scatter_panels()