import numpy as np
import networkx as nx
import BCR
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
# ===============================
# Settings
# ===============================

Gs = ['iceland','ca-sandi_auths','soc-karate','contiguous-usa']
NetLabels = ['Iceland','Sandia National Labs', 'Social Karate Club','USA Contiguous Border']
delta_vals = np.array([-2,-1,0,1,2])
c = 1
cmap = "coolwarm"
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 1.2
})
# ===============================
# Figure
# ===============================

fig, axes = plt.subplots(len(Gs),5, figsize=(18,14))

for row,GNAME in enumerate(Gs):

    # ---- load network ----
    G = BCR.open_graph(GNAME)
    adjm = BCR.adjacency(G)
    deg = np.sum(adjm,axis=1)
    P = adjm/deg[:,None]

    eta = BCR.solve_eta_matrix_exactly(adjm)
    alpha = BCR.conformist_label([[]],G.number_of_nodes())
    eta11,eta12,zeta1 = BCR.calculate_metrics_with_eta(adjm,alpha,eta)
    bc_star = eta11/(eta12-zeta1)

    # ---- SCC for each Δ ----
    gammas=[]
    for d in delta_vals:
        gamma,_ = BCR.calculate_gamma(P,eta,bc_star+d,c)
        gammas.append(gamma)
    gammas=np.array(gammas)

    # ---- responsiveness for size ----
    resp = BCR.calculate_gamma(P,eta,bc_star,c)[1]
    resp=(resp-resp.min())/(resp.ptp()+1e-9)
    sizes=60+300*resp

    # ---- layout ----
    pos = nx.spring_layout(G,seed=3)

    vmin,vmax=gammas.min(),gammas.max()

    for col in range(5):

        ax = axes[row,col]

        # edges (visible background skeleton)
        nx.draw_networkx_edges(
            G,pos,ax=ax,
            alpha=0.4,
            width=0.6,
            edge_color="0.6"
        )

        # normal nodes (semi-transparent)
        sc = nx.draw_networkx_nodes(
            G,pos,ax=ax,
            node_size=sizes,
            node_color=gammas[col],
            cmap=cmap,
            vmin=vmin,vmax=vmax,
            alpha=0.8,
            linewidths=0
        )

        # highlight top SCC nodes
        top=np.argsort(gammas[col])[-3:]
        nx.draw_networkx_nodes(
            G,pos,ax=ax,nodelist=top,
            node_size=sizes[top]*1.4,
            edgecolors='black',
            linewidths=2,
            node_color=gammas[col][top],
            cmap=cmap,
            vmin=vmin,vmax=vmax,
            alpha=1.0
        )

        # Δ labels only on first row
        if row==0:
            ax.set_title([r"$-2$",r"$-1$",r"$0$",r"$+1$",r"$+2$"][col],fontsize=18,fontfamily="serif")

        # network name on left
        if col==0:

            # panel label (a)-(d)
            ax.text(-0.45,1.05,f"{chr(ord('a')+row)}",
                    transform=ax.transAxes,
                    fontsize=23,fontweight='bold',
                    va='bottom', ha='right')
        
            # network name
            ax.text(-0.25,0.5,NetLabels[Gs.index(GNAME)],
                    rotation=90,
                    transform=ax.transAxes,
                    va='center',fontsize=18)

        # mark critical column
        '''if col==2:
            ax.text(0.5,-0.08,"critical point",
                    transform=ax.transAxes,
                    ha='center',fontsize=10)'''
        ax.axis("off")

# ===============================
# Layout adjustments
# ===============================

fig.subplots_adjust(right=0.9,hspace=0.05,wspace=0.05)

# ===============================
# Shared colorbar
# ===============================

cax=fig.add_axes([0.92,0.2,0.015,0.6])
cb=fig.colorbar(sc,cax=cax)
cb.set_label("Strategic Cooperation Contribution (SCC)",fontsize=18)

# ===============================
# Global Δ arrow
# ===============================
'''
arrow = FancyArrowPatch(
    (0.18,0.96), (0.82,0.96),
    transform=fig.transFigure,
    arrowstyle='->',
    linewidth=2,
    color='black'
)

fig.patches.append(arrow)'''

#fig.text(0.5,0.965,r"Increasing $\Delta$",ha="center",fontsize=14)

# ===============================
# Global Δ definition
# ===============================

fig.text(0.5,0.08,r"$\Delta=(b/c)-(b/c)^*$",ha="center",fontsize=18)

# ===============================
# Save
# ===============================

plt.savefig("./saves/imgs/SCC_Evolution.pdf",
            dpi=600,bbox_inches="tight")
plt.show()