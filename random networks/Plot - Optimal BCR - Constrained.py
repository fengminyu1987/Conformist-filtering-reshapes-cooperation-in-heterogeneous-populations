# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 21:05:26 2026

@author: ziyan
"""
# -*- coding: utf-8 -*-
"""
生成放入 SI 的随机网络无约束优化结果散点图 (含 y=x 对角线)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import os

def plot_si_synthetic_scatter(data_dir="."):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 13,
        'axes.linewidth': 1.2
    })

    # Nature 风格高级配色
    networks_info1 = [
        {'file': 'BA_0.1.csv', 'color': '#E64B35', 'label': 'Barabasi-Albert'},
        {'file': 'WS_0.1.csv', 'color': '#4DBBD5', 'label': 'Watts-Strogatz'},
        {'file': 'HK_0.1.csv', 'color': '#00A087', 'label': 'Holme-Kim'}
    ]

    ax1, ax2 = axes[0], axes[1]
    
    handles_list = []
    labels_list = []

    # --- 1. 绘制散点 ---
    for info in networks_info1:
        file_path = os.path.join(data_dir, info['file'])
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path)
        
        scatter_kwargs = {'s': 4, 'alpha': 0.25, 'c': info['color'], 'edgecolors': 'none'}
        
        scatter = ax1.scatter(df['initial_bcr'], df['initial_bcr']-df['optimal_bcr'], label=info['label'], **scatter_kwargs)
        
        handles_list.append(scatter)
        labels_list.append(info['label'])
    networks_info2 = [
        {'file': 'BA_0.2.csv', 'color': '#E64B35', 'label': 'Barabási-Albert'},
        {'file': 'WS_0.2.csv', 'color': '#4DBBD5', 'label': 'Watts-Strogatz'},
        {'file': 'HK_0.2.csv', 'color': '#00A087', 'label': 'Holme-Kim'}
    ]
    
    handles_list = []
    labels_list = []

    # --- 1. 绘制散点 ---
    for info in networks_info2:
        file_path = os.path.join(data_dir, info['file'])
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path)
        
        scatter_kwargs = {'s': 4, 'alpha': 0.25, 'c': info['color'], 'edgecolors': 'none'}
        
        scatter = ax2.scatter(df['initial_bcr'], df['initial_bcr']-df['optimal_bcr'], label=info['label'], **scatter_kwargs)
    
        
        handles_list.append(scatter)
        labels_list.append(info['label'])

    # --- 2. 核心修改：在图 a 加入 y = x 对角线 ---
    # 先获取当前散点画完后的坐标轴范围
    x_lim = ax1.get_xlim()
    y_lim = ax1.get_ylim()
    
    # 找到跨越 x 和 y 轴的最大和最小范围
    min_val = min(x_lim[0], y_lim[0])
    max_val = max(x_lim[1], y_lim[1])
    
    # 绘制对角线 (zorder=0 保证线在散点的最底层，不遮挡数据)
    line, = ax1.plot([min_val, max_val], [min_val, max_val], color='gray', 
                     linestyle='--', linewidth=1.5, zorder=0)
    
    # 极其重要：强制把坐标轴范围设回原本散点的范围，防止对角线拉长产生大量白边
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    
    # 将对角线加入图例
    handles_list.append(line)
    #labels_list.append(r'$(b/c)^* = (b/c)_{initial}$')
    
    #在图b也加入y = x对角线
    x_lim = ax2.get_xlim()
    y_lim = ax2.get_ylim()
    
    # 找到跨越 x 和 y 轴的最大和最小范围
    min_val = min(x_lim[0], y_lim[0])
    max_val = max(x_lim[1], y_lim[1])
    
    # 绘制对角线 (zorder=0 保证线在散点的最底层，不遮挡数据)
    line, = ax2.plot([min_val, max_val], [min_val, max_val], color='gray', 
                     linestyle='--', linewidth=1.5, zorder=0)
    
    # 极其重要：强制把坐标轴范围设回原本散点的范围，防止对角线拉长产生大量白边
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    # --- 3. 细节美化 ---
    ax1.text(-0.1, 1.05, "a", transform=ax1.transAxes, fontsize=18, fontweight='bold', va='bottom', ha='right')
    ax1.set_xlabel(r"$(b/c)^*_{ori}$")
    ax1.set_ylabel(r"$(b/c)^*_{opt}$ - $(b/c)^*_{ori}$")
    ax2.set_xlabel(r"$(b/c)^*_{ori}$")
    ax1.text(0.5, 0.65, "10% Conformists",transform=ax1.transAxes, fontsize=18, fontweight='bold', va='bottom', ha='right')
    ax2.text(0.5, 0.65, "20% Conformists",transform=ax2.transAxes, fontsize=18, fontweight='bold', va='bottom', ha='right')
    
    # 覆盖图例的透明度和大小
    leg = ax1.legend(handles_list, labels_list, loc='upper left', frameon=False, fontsize=11)
    for lh in leg.legend_handles[:]: # 除了最后一条线，其他散点恢复不透明度
        lh.set_alpha(1.0)
        lh.set_sizes([40])

    ax2.text(-0.1, 1.05, "b", transform=ax2.transAxes, fontsize=18, fontweight='bold', va='bottom', ha='right')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    save_path = "Synthetic_Networks.pdf"
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    print(f"绘图完成！图片已保存至: {save_path}")
    plt.show()

plot_si_synthetic_scatter(data_dir=".\\saves\\")