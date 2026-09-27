import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
import BCR

def generate_mechanism_figure(Graphs, NetLabels):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.linewidth': 1.2
    })
    fig, axes = plt.subplots(1, 5, figsize=(20,4), dpi=500)
    axes = axes.flatten()

    cmap = cm.get_cmap('plasma')
    panel_labels = ['a', 'b', 'c', 'd', 'e']

    all_scatter = []   # 用于共享 colorbar

    for i, GNAME in enumerate(Graphs):

        ax = axes[i]
        label = NetLabels[i]

        # ---------------------------
        # 1. 数据计算
        # ---------------------------
        G = BCR.open_graph(GNAME)
        n = G.number_of_nodes()
        nodes_list = list(G.nodes())

        adjm = BCR.adjacency(G)
        k = np.sum(adjm, axis=1).flatten()

        knn_dict = nx.average_neighbor_degree(G)
        knn = np.array([knn_dict[node] for node in nodes_list])

        alpha = BCR.conformist_label([[]], n)
        P = adjm / k[:, np.newaxis]
        eta = BCR.solve_eta_matrix_exactly(adjm)

        eta11, eta12, zeta1 = BCR.calculate_metrics_with_eta(adjm, alpha, eta)
        bc_star = eta11 / (eta12 - zeta1)

        gamma, R = BCR.calculate_gamma(P, eta, bc_star, 1.0)
        R = np.asarray(R).flatten()

        # ---------------------------
        # 2. Top 5% 节点
        # ---------------------------
        top_fraction = 0.05
        targets = np.argsort(R)[::-1][:int(n * top_fraction)]

        # ---------------------------
        # 3. 绘图
        # ---------------------------
        norm = colors.Normalize(vmin=np.min(R), vmax=np.max(R))

        sc = ax.scatter(
            k, knn,
            c=R, cmap=cmap, norm=norm,
            s=28, alpha=0.85,
            edgecolors='none',
            zorder=1
        )
        all_scatter.append(sc)

        # Top 5%：更明显的圈
        label_targets = "Top 5% Responders" if i == 0 else None
        ax.scatter(
            k[targets], knn[targets],
            s=70, facecolors='none',
            edgecolors='black',
            linewidths=1.4,
            zorder=2,
            label=label_targets
        )

        # ---------------------------
        # 4. 坐标轴与刻度
        # ---------------------------
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(r"Degree $k_i$", fontsize=16)
        if i == 0:
            ax.set_ylabel(r"Average Neighbor Degree $k_{nn,i}$", fontsize=16)

        ax.tick_params(
            axis='both', which='major',
            direction='in', length=6, width=1.5,
            labelsize=13, top=True, right=True
        )
        ax.minorticks_on()

        # 极淡 grid（PNAS 风格）
        ax.grid(True, which="major", ls=":", alpha=0.15)

        # ---------------------------
        # 5. 文本标注
        # ---------------------------
        # 子图编号：外侧左上
        ax.text(
            -0.14, 1.12, panel_labels[i],
            transform=ax.transAxes,
            fontsize=20, fontweight='bold',
            va='top', ha='left'
        )

        # 网络名：内部左上
        ax.text(
            0.4, 0.95, label,
            transform=ax.transAxes,
            fontsize=11,
            va='top', ha='left',fontfamily="serif",
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none')
        )

        # 图例仅放在 (a)
        if i == 0:
            ax.legend(
                loc='lower right',
                frameon=False,
                fontsize=10
            )

    # ---------------------------
    # 6. 共享 colorbar
    # ---------------------------
    fig.subplots_adjust(right=0.88, wspace=0.25, hspace=0.25)

    cax = fig.add_axes([0.90, 0.10, 0.015, 0.80])
    cbar = fig.colorbar(all_scatter[0], cax=cax)
    cbar.set_label(r"Responsiveness $R_i$", fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    plt.savefig(".\\saves\\imgs\\k_knn_r.pdf", bbox_inches='tight',dpi=500)
    plt.show()
Graphs = ['petster-hamster-household','arenas-email', 'web-polblogs','bio-CE-LC', 'power-eris1176']
NetLabels = ["Hamster Owners", "Email", "PolBlogs", "WormNet","Power Grid"]
generate_mechanism_figure(Graphs, NetLabels)