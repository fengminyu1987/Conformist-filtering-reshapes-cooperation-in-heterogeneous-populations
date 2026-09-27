# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 18:26:04 2026

@author: ziyan
"""

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import networkx as nx
import random
import BCR
import matplotlib.pyplot as plt
Graphs=['caenorhabditis-elegans', 'web-polblogs','petster-hamster-household','arenas-email']
GNAME="bio-CE-LC"
G=BCR.open_graph(GNAME)
n = G.number_of_nodes()
conformist_list=[random.sample(list(G.nodes()), 0)]
alpha=BCR.conformist_label(conformist_list,n)
adjm=BCR.adjacency(G)
degrees = np.sum(adjm, axis=1)
knn=list(nx.average_neighbor_degree(G).values())
P = adjm / degrees[:, np.newaxis]
eta=BCR.solve_eta_matrix_exactly(adjm)
eta11,eta12,zeta1=BCR.calculate_metrics_with_eta(adjm, alpha, eta)
bc=eta11/(eta12-zeta1)
b=15
colors=['red','green','blue']
c=1
gamma,responsiveness=BCR.calculate_gamma(P, eta, b, c)
plt.figure(figsize=(8,8))
plt.scatter(degrees,gamma,color='blue')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('$k_i$', fontsize=25)
plt.ylabel('$\\gamma_i$',fontsize=25)
plt.show()
plt.figure(figsize=(8,8))
plt.scatter(degrees,responsiveness/degrees,color='blue')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('$k_i$', fontsize=25)
plt.ylabel('$R_i$',fontsize=25)
plt.show()
plt.figure(figsize=(8,8))
plt.scatter(knn,gamma,color='red')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('$k_{nn,i}$', fontsize=25)
plt.ylabel('$\\gamma_i$',fontsize=25)
plt.show()
plt.figure(figsize=(8,8))
plt.scatter(knn,responsiveness/degrees,color='red')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('$k_{nn,i}$', fontsize=25)
plt.ylabel('$R_i$',fontsize=25)
plt.show()