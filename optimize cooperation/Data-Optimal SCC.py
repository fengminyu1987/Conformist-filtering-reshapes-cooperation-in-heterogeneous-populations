# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 23:28:02 2026

@author: ziyan
"""

import BCR
import networkx as nx
import numpy as np
#preparation
Graphs = ['petster-hamster-household','arenas-email', 'web-polblogs','bio-CE-LC','power-eris1176']
GNAME='bio-CE-LC'
G = BCR.open_graph(GNAME)
n = G.number_of_nodes()
nodes_list = list(G.nodes())
adjm = BCR.adjacency(G)
k = np.sum(adjm, axis=1).flatten()
alpha = BCR.conformist_label([[]], n)
P = adjm / k[:, np.newaxis]
eta = BCR.solve_eta_matrix_exactly(adjm)
eta11, eta12, zeta1 = BCR.calculate_metrics_with_eta(adjm, alpha, eta)
bc_star = eta11 / (eta12 - zeta1)
print(bc_star,GNAME)
#computing \gamma_i<0
bs=np.linspace(0.4*bc_star,2*bc_star,50)
#xaxis=np.linspace(0.4,2,50)
yaxis=[]
c=1
for b in bs:
    gamma, R = BCR.calculate_gamma(P, eta, b, c)
    yaxis.append(np.sum(gamma<0)/n)
#np.savetxt(".\\saves\\x_optimal_SCC.txt",xaxis,fmt='%.8f', delimiter=',')
np.savetxt(".\\saves\\y_{}_SCC.txt".format(GNAME),yaxis,fmt='%.8f', delimiter=',')