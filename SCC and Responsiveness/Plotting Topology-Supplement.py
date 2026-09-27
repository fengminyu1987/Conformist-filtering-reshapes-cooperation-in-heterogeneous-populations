import numpy as np
import matplotlib.pyplot as plt
import BCR
import networkx as nx
import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

Graphs = ['petster-hamster-household', 'arenas-email', 'web-polblogs', 'bio-CE-LC', 'power-eris1176']
NetLabels = ["Hamster Owners", "Email", "PolBlogs", "WormNet", "Power Grid"]
#Graphs = ['dolphins', 'dolphins', 'dolphins', 'dolphins', 'dolphins'] test
# 创建 3行 (b的变化) x 5列 (不同的网络) 的画板
# 稍微增加了高度以便在底部放置 colorbar
fig, axes = plt.subplots(3, 5, figsize=(22, 14))

panel_idx = 0

# 定义图内的文本标签
b_labels = [r"$(b/c)^*-1$", r"$(b/c)^*$", r"$(b/c)^*+1$"]
b_offsets = [-1, 0, 1]

for col, GNAME in enumerate(Graphs):
    # 1. 针对该网络，计算基础结构信息
    print("Processing {}...".format(GNAME))
    G = BCR.open_graph(GNAME)
    n = G.number_of_nodes()

    alpha = BCR.conformist_label([[]], n)
    adjm = BCR.adjacency(G)

    k = np.sum(adjm, axis=1)
    knn = np.array(list(nx.average_neighbor_degree(G).values()))

    P = adjm / k[:, np.newaxis]
    eta = BCR.solve_eta_matrix_exactly(adjm)
    eta11, eta12, zeta1 = BCR.calculate_metrics_with_eta(adjm, alpha, eta)

    bc = eta11 / (eta12 - zeta1)
    c = 1

    k = np.asarray(k)
    knn = np.asarray(knn)

    # ---------- size from knn ----------
    knn_clip = np.clip(knn, np.percentile(knn, 5), np.percentile(knn, 95))
    knn_norm = (knn_clip - knn_clip.min()) / (knn_clip.ptp() + 1e-12)
    sizes = 40 + 400 * knn_norm

    # 2. 提前计算该网络在 3 个不同 b 值下的 gamma，以便求出统一的颜色范围 (vmin, vmax)
    gammas_for_this_net = []
    Rs_for_this_net = []
    for offset in b_offsets:
        b = bc + offset
        gamma, R = BCR.calculate_gamma(P, eta, b, c)
        gammas_for_this_net.append(np.asarray(gamma))
        Rs_for_this_net.append(np.asarray(R))
        
    # 确定当前网络（当前列）的颜色上下限
    vmin = np.min(gammas_for_this_net)
    vmax = np.max(gammas_for_this_net)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # 3. 开始绘制该列的 3 个子图
    for row, offset in enumerate(b_offsets):
        ax = axes[row, col]
        
        gamma = gammas_for_this_net[row]
        R = Rs_for_this_net[row]

        # ---------- scatter (individual nodes) ----------
        sc = ax.scatter(
            k, R,
            c=gamma,
            s=sizes,
            cmap="coolwarm",
            norm=norm, # 使用该列统一的颜色归一化标准
            alpha=0.85,
            linewidths=0,
            zorder=1
        )

        # ---------- labels ----------
        if row == 2:  # 仅在最底部的子图显示 x 轴标签
            ax.set_xlabel(r"$k_i$", fontsize=18, fontfamily="serif")
        if col == 0:  # 仅在最左侧的子图显示 y 轴标签
            ax.set_ylabel(r"$R_i$", fontsize=18, fontfamily="serif")

        # PNAS/Nature 风格的 Panel Label: 粗体、无括号，放在左上角
        panel_letter = chr(ord('a') + panel_idx)
        ax.text(
            -0.12, 1.05, f"{panel_letter}",
            transform=ax.transAxes,
            fontsize=20, fontweight='bold', fontfamily="Arial", # Nature通常偏好无衬线字体作为Panel编号
            ha='right', va='bottom'
        )
        panel_idx += 1

        # 列标题 (在第一行顶部显示网络名称)
        if row == 0:
            ax.set_title(NetLabels[col], fontsize=18, fontfamily="serif", pad=15)

        # 子图内的文本框，仅显示当前的 b 值公式 (去掉网络名称)
        ax.text(
            0.05, 0.95, b_labels[row],
            transform=ax.transAxes,
            va='top', ha='left', fontfamily="serif",
            fontweight='normal', fontsize=14,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
        )

        # ticks 样式美化
        ax.tick_params(axis='both', which='major', direction='in',
                       length=6, width=1.5, top=True, right=True, labelsize=12)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='in',
                       length=3, width=1, top=True, right=True)

        ax.grid(alpha=0.15, linestyle=':')

        # ---------- size legend only in first panel ----------
        if row == 0 and col == 0:
            for q, label in zip([0.2, 0.5, 0.8], ["low", "medium", "high"]):
                s = 80 + 345 * q
                ax.scatter([], [], s=s, c="gray", alpha=0.6, label=label)

            ax.legend(title=r"$k_{nn,i}$",
                      frameon=False, loc="best", fontsize=12, title_fontsize=13)
                      
        ax.set_xscale("log")
        # 绘制 R_i 的 80% 分位数参考线
        ax.axhline(np.percentile(R, 80), color="0.6", ls=":", lw=1.5)

    # 4. 为当前列（当前网络）在底部添加专属的 Colorbar
    bottom_ax = axes[-1, col]

    # 强行在 bottom_ax 下方创建一个宽度与之一致的坐标系 (cax)
    # width="100%" 保证与子图绝对等宽
    # height="8%" 控制 Colorbar 的粗细
    # bbox_to_anchor 的第二个参数 (这里是 -0.45) 控制向下偏移的绝对距离，可微调
    cax = inset_axes(bottom_ax,
                     width="100%", 
                     height="8%", 
                     loc='lower center',
                     bbox_to_anchor=(0, -0.45, 1, 1), 
                     bbox_transform=bottom_ax.transAxes,
                     borderpad=0)

    # 将 colorbar 画在我们刚建好的完美对齐的 cax 中
    cbar = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cbar.solids.set_rasterized(True)
    # 调整文字和刻度
    cbar.set_label(r"Contribution $\gamma_i$", fontsize=14, fontfamily="serif", labelpad=8)
    cbar.ax.tick_params(labelsize=11)

# 调整子图之间的间距
fig.subplots_adjust(wspace=0.25, hspace=0.25, bottom=0.22) # 给底部的colorbar留出足够空间fig.subplots_adjust(wspace=0.25, hspace=0.25, bottom=0.22)

plt.savefig("./saves/imgs/k_R_gamma_knn_sitest.png", dpi=500, bbox_inches="tight")
plt.show()