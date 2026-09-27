# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 05:27:02 2026

@author: ziyan
"""

import BCR
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
def Generate_Joined_Complete(N):
    g1=nx.complete_graph(N)
    g2=nx.complete_graph(N)
    G=nx.disjoint_union(g1,g2)
    G.add_edge(0,N)
    return G
def gamma_theory_u(N,B,C):
    return -(
    (N - 2) * (
        C * N**2 * (N**4 + N**2 - N + 1)
        + B * (N**5 - 3*N + 2)
    )
) / (
    2 * N**2 * (N**5 + 2*N**3 - 3*N**2 + 4*N - 2)
)
def bcr(N, Mu, Mu_star):
    UC=N*N*N*(N-2)*(N*N*N*N+N*N-N+1)*Mu
    USC=N*N*(N-1)*(2*N*N*N*N*N+N*N-3*N+2)*Mu_star
    UB=N*(N-2)*(-N*N*N*N*N+3*N-2)*Mu
    USB=(N-1)*(N-1)*(N-1)*(N*N*N*N*N+N*N*N*N+N*N*N+2*N*N-N+2)*Mu_star
    return (UC+USC)/(UB+USB)
def gamma_theory_us(N,B,C):
    return (
    (N - 1) * (
        B * (N - 1)**2 * (N**5 + N**4 + N**3 + 2*N**2 - N + 2)
        - C * N**2 * (2*N**5 + N**2 - 3*N + 2)
    )
) / (
    2 * N**3 * (N**5 + 2*N**3 - 3*N**2 + 4*N - 2)
)
def R_theory_u(N):
    return -(N - 2)*(N**5 - 3*N + 2)/(2 * N**2 * (N**5 + 2*N**3 - 3*N**2 + 4*N - 2))
def R_theory_us(N):
    return (N-1)**3*(N**5 + N**4 + N**3 + 2*N**2 - N + 2)/(2*N**3*(N**5 + 2*N**3 - 3*N**2 + 4*N - 2))