# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 02:20:11 2026

@author: ziyan
"""

import BCR
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import networkx as nx
import time
import matplotlib.pyplot as plt
import random
def Generate_Star_HH(N):
    g1=nx.star_graph(N-1)
    g2=nx.star_graph(N-1)
    G=nx.disjoint_union(g1,g2)
    G.add_edge(0,N)
    return G
def gamma_theory(N,B,C):
    return ((N - 1) * (
        C * N**2 * (-10*N**2 + 3*N + 2)
        + B * (4*N**4 - 4*N**3 - N**2 + 3*N - 2)
    )) / (2 * N**3 * (4*N**2 + 4*N - 3))
'''
N=15
b=5
c=1
G = Generate_Star_HH(N)
n = G.number_of_nodes()
conformist_list=[]#random.sample(list(G.nodes()), 99)
alpha=BCR.conformist_label(conformist_list,n)
adjm=BCR.adjacency(G)
degrees = np.sum(adjm, axis=1)
P = adjm / degrees[:, np.newaxis]
eta=BCR.solve_eta_matrix_exactly(adjm)
eta11,eta12,zeta1=BCR.calculate_metrics_with_eta(adjm, alpha, eta)
bc=eta11/(eta12-zeta1)
gamma=BCR.calculate_gamma(P, eta, b, c)
gammaT=gamma_theory(N,b,c)'''
N=15
c=1
G = Generate_Star_HH(N)
bs=np.linspace(1,10,200)
n = G.number_of_nodes()
conformist_list=[]#random.sample(list(G.nodes()), 99)
alpha=BCR.conformist_label(conformist_list,n)
adjm=BCR.adjacency(G)
degrees = np.sum(adjm, axis=1)
P = adjm / degrees[:, np.newaxis]
eta=BCR.solve_eta_matrix_exactly(adjm)
eta11,eta12,zeta1=BCR.calculate_metrics_with_eta(adjm, alpha, eta)
bc=eta11/(eta12-zeta1)
plt.figure(figsize=(8,8))
gamma=gamma_theory(N,bs,c)
plt.plot(bs,gamma, color='red', lw=3,label='$\gamma_h$')
plt.legend(fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid()
plt.tick_params(axis='both', # 应用于x轴和y轴
which='major', # 应用于主刻度线
direction='in', # 刻度线向内外都显示
length=6, # 刻度线长度
width=5, # 刻度线宽度
colors='black', # 刻度线颜色
pad=10, # 刻度线与标签的距离
labelsize=20, # 标签字体大小
labelcolor='black', # 标签颜色
bottom=True, # 显示x轴的底部刻度线
top=True, # 不显示x轴的顶部刻度线
left=True, # 显示y轴的左侧刻度线
right=True) # 不显示y轴的右侧刻度线
plt.ylabel('SCC', fontsize=25)
plt.xlabel('$b$', fontsize=25)
plt.show()