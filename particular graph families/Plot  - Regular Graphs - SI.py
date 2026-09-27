# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 21:13:37 2026

@author: ziyan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_nature_style_rrg_results():
    # ==========================================
    # 1. 顶刊全局画图风格设置 (Nature/PNAS 风格)
    # ==========================================
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,             # 基础字体大小
        "axes.titlesize": 10,       # 标题字体大小
        "axes.labelsize": 9,        # 轴标签字体大小
        "xtick.labelsize": 8,       # 刻度线标签大小
        "ytick.labelsize": 8,
        "axes.linewidth": 1.0,      # 坐标轴线宽
        "xtick.major.width": 1.0,   # 刻度线粗细
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,         # 确保保存PDF时字体可编辑
        "ps.fonttype": 42
    })

    # ==========================================
    # 2. 准备画布 (双栏排版宽度通常约为 180mm / 7.2 inch)
    # ==========================================
    degrees = [4, 6, 8, 10]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), dpi=300)
    axes = axes.flatten()

    # 设定一种主色调，这里选用一种沉稳的科学蓝/青色
    main_color = "#1F77B4" 

    print("开始绘制图表...")

    # ==========================================
    # 3. 循环读取数据并绘制每个子图
    # ==========================================
    for i, k in enumerate(degrees):
        ax = axes[i]
        filename = f".\\saves\\RRG_bc_ratio_k{k}.csv"
        
        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"警告: 找不到文件 {filename}，跳过此图。请确保上一步的数据已生成。")
            continue
            
        # 读取数据
        df = pd.read_csv(filename)
        
        # 将宽表 (Wide form) 转换为 Seaborn 喜欢的长表 (Long form)
        df_melted = df.melt(var_name="Proportion", value_name="BC_Ratio")
        
        # 整理 x 轴标签 (把 'p_0.1' 变成 '0.1')
        df_melted["Proportion"] = df_melted["Proportion"].str.replace("p_", "").astype(float)

        # ------------------------------------------
        # 绘图层 1：底层散点 (Strip plot)
        # 因为有1000个点，我们设置极小的尺寸和高透明度(alpha)
        # ------------------------------------------
        sns.stripplot(
            data=df_melted, 
            x="Proportion", 
            y="BC_Ratio", 
            ax=ax,
            color=main_color,
            alpha=0.15,      # 透明度：展现数据密度
            size=2.0,        # 极小的散点
            jitter=0.25,     # 抖动范围
            zorder=1         # 置于底层
        )

        # ------------------------------------------
        # 绘图层 2：顶层箱线图 (Box plot)
        # ------------------------------------------
        sns.boxplot(
            data=df_melted, 
            x="Proportion", 
            y="BC_Ratio", 
            ax=ax,
            color="white",        # 箱体填充白色
            linewidth=1.2,        # 箱线加粗
            fliersize=0,          # 隐藏箱线图自带的异常点（防止和底层散点重复）
            width=0.5,            # 箱体宽度
            zorder=2,             # 置于顶层
            boxprops=dict(edgecolor='black', alpha=0.8), # 箱体边框
            medianprops=dict(color='black', linewidth=1.5), # 中位数线加粗加黑
            whiskerprops=dict(color='black', linewidth=1.2),
            capprops=dict(color='black', linewidth=1.2)
        )

        # ------------------------------------------
        # 细节修饰：坐标轴与标签
        # ------------------------------------------
        ax.text(0.1,0.9,"$k={:d}$".format(k),transform=ax.transAxes, 
                    fontsize=11, fontweight='bold')
        ax.axhline(98/(100/k-2),color='red',ls='--',lw=2.5,alpha=0.6,zorder=3)
        # 仅在底部的图显示 x 轴标签，左侧的图显示 y 轴标签，使图表更清爽
        if i >= 2:
            ax.set_xlabel("Proportion of Conformists", fontsize=9)
        else:
            ax.set_xlabel("")
            
        if i % 2 == 0:
            ax.set_ylabel("Critical Ratio $(b/c)^*$", fontsize=9)
        else:
            ax.set_ylabel("")

        # 移除上方和右侧的边框线 (Despine)
        sns.despine(ax=ax)
        
        # 添加面板字母标签 (a), (b), (c), (d) - 顶刊必备
        panel_label = chr(97 + i)  # 97 是 'a' 的 ASCII 码
        ax.text(-0.15, 1.05, f"{panel_label}", transform=ax.transAxes, 
                fontsize=11, fontweight='bold', va='top', ha='right')

    # ==========================================
    # 4. 整体布局调整与保存
    # ==========================================
    plt.tight_layout(pad=1.5)
    
    # 保存为高分辨率 PDF (矢量图，推荐投稿使用) 和 PNG
    plt.savefig(".\\saves\\imgs\\RRG.pdf", format='pdf',dpi=500, bbox_inches='tight')
    #plt.savefig("RRG_Conformists_Boxplot_NatureStyle.png", dpi=300, bbox_inches='tight')
    plt.show()

# 运行绘图代码
if __name__ == "__main__":
    plot_nature_style_rrg_results()