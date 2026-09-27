# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 22:29:23 2026

@author: ziyan
"""

import matplotlib.pyplot as plt
import numpy as np

import networkx as nx

import BCR

import numpy as np
G=nx.generators.empty_graph()
b=17
c=1
G.add_edges_from([(0,1),(0,2),(1,3),(2,4),(2,5),(3,4),(3,5),(3,7),(4,5),(4,6),(5,6),(5,7),(6,7)])

N= G.number_of_nodes()

adjm=BCR.adjacency(G)

degrees = np.sum(adjm, axis=1)

P = adjm / degrees[:, np.newaxis]

eta=BCR.solve_eta_matrix_exactly(adjm)

Ci,Ri=BCR.calculate_CR(P, eta)
alpha,lam=BCR.solve_minimal_ST(Ci, Ri, 3, max_iter=100, tol=1e-6)
gamma,responsiveness=BCR.calculate_gamma(P, eta, b, c)
node_order = [6, 3, 1, 4, 7, 2, 5, 0]

# 按照该顺序提取数据
gamma_sorted = gamma[node_order]
R_sorted = responsiveness[node_order]
y_labels = [str(i) for i in node_order] # Y轴显示的标签

# 绘图风格设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.0

# 颜色定义
c_pos = '#E64B35' # 红 (Gamma > 0)
c_neg = '#4DBBD5' # 蓝 (Gamma < 0)
c_R   = '#7E6148' # 褐/灰 (Ri)

# ------------------------------------------------------
# 图 1: Gamma 分布图 (Strategic Contribution)
# ------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(3.5, 7.5)) # 宽:高 = 1:3

# 生成颜色列表
colors = [c_pos if val <= 0 else c_neg for val in gamma_sorted]
colors[-1]='grey'
colors[-2]='grey'
colorsR= [c_pos if val <= 0 else c_neg for val in R_sorted]
colorsR[-1]='grey'
colorsR[-2]='grey'
# 绘制水平条形
# y_pos 使用 range(len)，配合后面的 set_yticks
y_positions = np.arange(len(node_order))
ax1.barh(y_positions, gamma_sorted, color=colors, height=0.6, align='center')

# 核心：设置 Y 轴刻度与标签
ax1.set_yticks([])
#ax1.set_yticklabels(y_labels)
ax1.set_xticks([])
# 核心：反转 Y 轴，使得 node_order 的第一个元素 (6) 显示在最上方
ax1.invert_yaxis()

# 装饰
ax1.axvline(0, color='black', linewidth=0.8) # 0 基准线
#ax1.set_xlabel(r"Contribution $\gamma_i$")
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False) # 隐藏左轴线，只留文字以便对齐

plt.tight_layout()
plt.savefig('Plot_Gamma_Aligned.png', transparent=True)
plt.show()

# ------------------------------------------------------
# 图 2: Responsiveness 分布图 (Ri)
# ------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(3.5, 7.5)) # 宽:高 = 1:3

ax2.barh(y_positions, R_sorted, color=colorsR, height=0.6, align='center')

# 同样的坐标轴处理
ax2.set_yticks([])
ax2.axvline(0, color='black', linewidth=0.8)
#ax2.set_yticklabels(y_labels)
ax2.invert_yaxis() # 6 在最上
ax2.set_xticks([])
# 装饰
#ax2.set_xlabel(r"Responsiveness $R_i$")
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('Plot_Ri_Aligned.png', transparent=True,dpi=500)
plt.show()