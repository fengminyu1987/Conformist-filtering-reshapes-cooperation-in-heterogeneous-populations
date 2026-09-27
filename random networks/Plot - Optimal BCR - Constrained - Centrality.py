# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 21:05:26 2026

@author: ziyan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Nature/PNAS 级图表参数设置 ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'figure.dpi': 300
})

def plot_main_gap_violin(ratios=[0.1, 0.2], networks=['BA', 'WS', 'HK']):
    heuristics = ['Degree', 'CI', 'K-Shell', 'Betweenness', 'Closeness', 'Eigenvector', 'PageRank']
    
    # 创建 2x1 的垂直画布 (a, b 两个面板)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    for idx, ratio in enumerate(ratios):
        ax = axes[idx]
        all_data = []
        
        # 1. 收集并合并该比例下的所有网络数据
        for net in networks:
            file_path = f"./saves/{net}_{ratio}_Centrality_Compare.csv"
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found. Skipping...")
                continue
                
            df = pd.read_csv(file_path)
            
            # 计算 Gap = Heuristic - Optimal
            for h in heuristics:
                df[f'Gap_{h}'] = df[h] - df['optimal_bcr']
                
            # Melt 宽表变长表，适配 Seaborn 画图
            melted = pd.melt(df, value_vars=[f'Gap_{h}' for h in heuristics], 
                             var_name='Heuristic', value_name='Gap')
            melted['Heuristic'] = melted['Heuristic'].str.replace('Gap_', '')
            melted['Network'] = net
            all_data.append(melted)
            
        if not all_data:
            continue
            
        df_plot = pd.concat(all_data, ignore_index=True)
        #ax.text(-0.1, 1.05, "a", transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom', ha='right')
        # 2. 绘制小提琴图
        # cut=0 保证小提琴图的尾巴不会画到数据实际范围之外（即不会画到 0 以下）
        sns.violinplot(data=df_plot, x='Network', y='Gap', hue='Heuristic', 
                       ax=ax, palette='Set2', inner='quartile', linewidth=0.8, cut=0)
        
        # 3. 画一条醒目的 Y=0 灵魂基准线
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, zorder=0, alpha=0.8)
        
        # 4. 图表细节美化
        ax.set_ylabel(r"Optimality Gap Compared with $(b/c)^*_{opt}$")
        ax.set_xlabel("")
        ax.set_title(f"Conformist Ratio $\\Phi = {ratio}$", loc='left', fontweight='bold')
        
        # 去除顶部和右侧的边框 (APA/Nature 常用风格)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.05, 1.05,f"{chr(ord('a')+idx)}",
                transform=ax.transAxes,fontsize=16,fontweight='bold',va='bottom', ha='right')
        # 处理图例 (只在第一个图显示即可)
        if idx == 0:
            ax.legend(title='Centrality Heuristics', ncol=7, loc='upper center', 
                      bbox_to_anchor=(0.5, 1.25))
        else:
            ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig('MainText_Optimality_Gap.pdf', format='pdf',dpi=500, bbox_inches='tight')
    plt.show()

# 执行画图
plot_main_gap_violin()