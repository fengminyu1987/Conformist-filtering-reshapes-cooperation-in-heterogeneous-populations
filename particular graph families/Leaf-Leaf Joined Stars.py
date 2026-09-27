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
def Generate_Star_LL(N):
    g1=nx.star_graph(N-1)
    g2=nx.star_graph(N-1)
    G=nx.disjoint_union(g1,g2)
    G.add_edge(1,N+1)
    return G
def gamma_theory_h(N,B,C):
    return (
    (N - 2) * (
        C * (-350*N**4 + 559*N**3 + 313*N**2 - 919*N + 397)
        + B * (70*N**4 - 29*N**3 - 232*N**2 + 247*N - 60)
    )
) / (
    2 * (N - 1) * (2*N - 1) * (70*N**3 - 7*N**2 + 12*N - 59)
)
def responsiveness_theory_h(N):
    return (N-2)*(70*N**4-29*N**3-232*N**2+247*N-60)/(2*(N-1)*(2*N-1)*(70*N**3-7*N**2+12*N-59))
def responsiveness_theory_l(N):
    return (210*N**5-491*N**4-202*N**3+1263*N**2-960*N+188)/(4*(N-1)*(2*N-1)*(70*N**3-7*N**2+12*N-59))
def gamma_theory_l(N,B,C):
    return (
    -4 * C * (N - 1)**2 * (70*N**3 + 97*N**2 - 175*N - 30)
    + B * (210*N**5 - 491*N**4 - 202*N**3 + 1263*N**2 - 960*N + 188)
) / (
    4 * (N - 1) * (2*N - 1) * (70*N**3 - 7*N**2 + 12*N - 59)
)
def bcr(N, nh, nl):
    HC=(N - 2)*(350* N**4-559*N**3-313* N**2+919* N-397)
    HB=(N - 2)*(70 *N**4-29*N**3-232*N**2 + 247*N - 60)
    LC=4 *((N - 1)**2) *(70 *N**3 + 97 *N**2 - 175* N - 30)
    LB=210 *N**5 - 491 *N**4 - 202* N**3 + 1263* N**2 - 960* N + 188
    return (2*nh*HC+nl*LC)/(2*nh*HB+nl*LB)
Ns=np.linspace(5,60,200)
plt.figure(figsize=(6,6))
plt.plot(Ns,responsiveness_theory_h(Ns),c='red',label="Hubs")
plt.plot(Ns,responsiveness_theory_l(Ns),c='blue',label="Connector Leaves")
plt.legend(fontsize=15)
plt.show()
plt.figure(figsize=(6,6))
plt.plot(Ns,bcr(Ns, 2, 2),c='green')
plt.ylim([0,10])
plt.show()