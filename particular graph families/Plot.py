import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import networkx as nx
import string

# ==========================================
# 1. 理论公式定义 (保持不变)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 1.2
})
def calc_k_regular(N, k=4):
    R = (N - 2*k) / (2 * k * N)
    denom = N/k - 2
    bc_crit = np.full_like(N, 9999.0) 
    valid_mask = denom > 0
    bc_crit[valid_mask] = (N[valid_mask] - 2) / denom[valid_mask]
    return R, bc_crit

def calc_center_connected_stars(N):
    num_R = (N - 1) * (4*N**4 - 4*N**3 - N**2 + 3*N - 2)
    den_R = 2 * N**3 * (4*N**2 + 4*N - 3)
    R_h = num_R / den_R
    R_l = np.zeros_like(N)
    
    term_c = N**2 * (10*N**2 - 3*N - 2)
    term_b = 4*N**4 - 4*N**3 - N**2 + 3*N - 2
    bc_crit = term_c / term_b
    return R_h, R_l, bc_crit

def calc_leaf_connected_stars(N):
    denom = 2 * (N - 1) * (2*N - 1) * (70*N**3 - 7*N**2 + 12*N - 59)
    num_Rh = (N - 2) * (70*N**4 - 29*N**3 - 232*N**2 + 247*N - 60)
    R_h = num_Rh / denom
    num_Rl_star = 210*N**5 - 491*N**4 - 202*N**3 + 1263*N**2 - 960*N + 188
    R_l_star = num_Rl_star / (2 * denom)
    
    Hub_Cb = (N - 2) * (70*N**4 - 29*N**3 - 232*N**2 + 247*N - 60) / denom
    Hub_Cc = (N - 2) * (350*N**4 - 559*N**3 - 313*N**2 + 919*N - 397) / denom
    Lstar_Cb = (210*N**5 - 491*N**4 - 202*N**3 + 1263*N**2 - 960*N + 188) / (2 * denom)
    Lstar_Cc = 4 * (N - 1)**2 * (70*N**3 + 97*N**2 - 175*N - 30) / (2 * denom)
    bc_crit = (Hub_Cc + Lstar_Cc) / (Hub_Cb + Lstar_Cb)
    return R_h, R_l_star, bc_crit

def calc_connected_cliques(N):
    denom = 2 * N**3 * (N**5 + 2*N**3 - 3*N**2 + 4*N - 2)
    num_Ru = (N - 2) * (-N**5 + 3*N - 2)
    R_u = num_Ru / denom
    num_Ru_star = (N - 1)**3 * (N**5 + N**4 + N**3 + 2*N**2 - N + 2)
    R_u_star = num_Ru_star / denom
    
    u_Cb = (N - 2) * (-N**5 + 3*N - 2) / denom 
    u_Cc = (N - 2) * (N**2 * (N**4 + N**2 - N + 1)) / denom 
    ustar_Cb = (N - 1)**3 * (N**5 + N**4 + N**3 + 2*N**2 - N + 2) / denom 
    ustar_Cc = (N - 1) * (N**2 * (2*N**5 + N**2 - 3*N + 2)) / denom
    
    Total_Cb = 2 * ustar_Cb + (2*N - 2) * u_Cb
    Total_Cc = 2 * ustar_Cc + (2*N - 2) * u_Cc
    bc_crit = Total_Cc / Total_Cb
    return R_u, R_u_star, bc_crit

# ==========================================
# 2. 网络生成函数
# ==========================================

def Generate_Lattice(dim=4):
    G = nx.grid_2d_graph(dim, dim, periodic=True)
    return G

def Generate_Star_LL(N):
    g1 = nx.star_graph(N-1)
    g2 = nx.star_graph(N-1)
    mapping = {i: i+N for i in g2.nodes()}
    g2 = nx.relabel_nodes(g2, mapping)
    G = nx.disjoint_union(g1, g2)
    G.add_edge(1, N+1) 
    return G

def Generate_Star_HH(N):
    g1 = nx.star_graph(N-1)
    g2 = nx.star_graph(N-1)
    mapping = {i: i+N for i in g2.nodes()}
    g2 = nx.relabel_nodes(g2, mapping)
    G = nx.disjoint_union(g1, g2)
    G.add_edge(0, N)
    return G

def Generate_Joined_Complete(N):
    g1 = nx.complete_graph(N)
    g2 = nx.complete_graph(N)
    mapping = {i: i+N for i in g2.nodes()}
    g2 = nx.relabel_nodes(g2, mapping)
    G = nx.disjoint_union(g1, g2)
    G.add_edge(0, N)
    return G

# ==========================================
# 3. 绘图主程序
# ==========================================

def draw_network_inset(ax, plot_type, color_map):
    """绘制小网络示意图"""
    N_demo = 6 
    
    if plot_type == 0: # Lattice
        G = Generate_Lattice(dim=4)
        pos = nx.kamada_kawai_layout(G)
        node_colors = ['black'] * len(G)
        edge_color = 'gray'
        
    elif plot_type == 1: # Star HH
        G = Generate_Star_HH(N_demo)
        pos = nx.spring_layout(G, seed=42, k=0.5)
        node_colors = []
        for n in G.nodes():
            if n == 0 or n == N_demo:
                node_colors.append(color_map['Hubs'])
            else:
                node_colors.append(color_map['Leaves'])
        edge_color = 'gray'

    elif plot_type == 2: # Star LL
        G = Generate_Star_LL(N_demo)
        pos = nx.spring_layout(G, seed=31, k=0.5) 
        node_colors = []
        for n in G.nodes():
            if n == 1 or n == N_demo + 1:
                node_colors.append(color_map['Connector'])
            elif n == 0 or n == N_demo:
                node_colors.append(color_map['Hubs'])
            else:
                node_colors.append('lightgray')
        edge_color = 'gray'
        
    elif plot_type == 3: # Cliques
        G = Generate_Joined_Complete(N_demo)
        pos = nx.spring_layout(G, seed=10, k=1.2)
        node_colors = []
        for n in G.nodes():
            if n == 0 or n == N_demo:
                node_colors.append(color_map['Connector'])
            else:
                node_colors.append(color_map['Bulk'])
        edge_color = 'gray'

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, node_color=node_colors, linewidths=0.7, edgecolors='white')
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.8, edge_color=edge_color, alpha=0.6)
    ax.axis('off')

def plot_final_with_insets_adjusted():
    c_red = '#E64B35'
    c_blue = '#3C5488'
    c_teal = '#00A087'
    c_light_blue = '#4DBBD5'
    
    N_range_A = np.linspace(5, 60, 300)
    N_range_B = np.linspace(3, 60, 300)
    
    plot_configs = []
    
    # 1. k-Regular
    r, bc = calc_k_regular(N_range_A, k=4)
    plot_configs.append({
        "type": 0,
        "title": "Vertex-Transitive\nk-Regular ($k=4$)",
        "N": N_range_A,
        "R_lines": [("All Nodes", r, 'black', '-')],
        "bc": bc,
        "ylim_bc": (1, 8),
        "colors": {},
        # (a) 曲线在底部，Inset 放右上角安全
        "inset_bounds": [0.6, 0.55, 0.35, 0.4] 
    })
    
    # 2. Center-Stars
    rh, rl, bc = calc_center_connected_stars(N_range_B)
    plot_configs.append({
        "type": 1,
        "title": "Center-Connected\nStars",
        "N": N_range_B,
        "R_lines": [("Hubs", rh, c_red, '-'), ("Leaves", rl, c_light_blue, '--')],
        "bc": bc,
        "ylim_bc": (2.5, 3.2),
        "colors": {'Hubs': c_red, 'Leaves': c_light_blue},
        # (b) Hub曲线(红)在顶部 ~0.5，Leaf曲线在底部 ~0。
        # Inset 放在中间偏下，避开红色高线
        "inset_bounds": [0.6, 0.15, 0.35, 0.4]
    })
    
    # 3. Leaf-Stars
    rh, rl_s, bc = calc_leaf_connected_stars(N_range_B)
    plot_configs.append({
        "type": 2,
        "title": "Leaf-Connected\nStars",
        "N": N_range_B,
        "R_lines": [("Connector Leaf", rl_s, c_red, '-'), ("Hubs", rh, c_teal, '-.')],
        "bc": bc,
        "ylim_bc": (2.6, 3.2),
        "colors": {'Connector': c_red, 'Hubs': c_teal},
        # (c) 曲线都在 ~0.3-0.4，顶部 0.5-0.6 是空的。
        # Inset 放右上角安全
        "inset_bounds": [0.6, 0.6, 0.35, 0.4]
    })
    
    # 4. Cliques
    ru, ru_s, bc = calc_connected_cliques(N_range_B)
    plot_configs.append({
        "type": 3,
        "title": "Connected\nCliques",
        "N": N_range_B,
        "R_lines": [("Connector", ru_s, c_red, '-'), ("Bulk Nodes", ru, c_blue, '--')],
        "bc": bc,
        "ylim_bc": (2.0, 5.0),
        "colors": {'Connector': c_red, 'Bulk': c_blue},
        # (d) Connector曲线(红)很高 ~0.5。
        # Inset 放在中间偏下，避开红色高线
        "inset_bounds": [0.6, 0.20, 0.35, 0.4]
    })
    
    # Setup Figure
    fig = plt.figure(figsize=(16, 8), dpi=500)
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1.2], hspace=0.15, wspace=0.25)
    
    for i, cfg in enumerate(plot_configs):
        ax_R = fig.add_subplot(gs[0, i])
        ax_BC = fig.add_subplot(gs[1, i])
        
        # --- 1. Draw Responsiveness Curves ---
        for label, y_vals, color, ls in cfg["R_lines"]:
            ax_R.plot(cfg["N"], y_vals, label=label, color=color, lw=2.5, linestyle=ls)
            
        ax_R.set_title(cfg["title"], fontsize=13, fontweight='bold', pad=12)
        ax_R.set_xticklabels([])
        ax_R.grid(True, linestyle=':', alpha=0.5)
        ax_R.set_ylim(-0.05, 0.65)
        panel_label = string.ascii_lowercase[i]
        ax_R.text(-0.05, 1.05, f"{panel_label}", transform=ax_R.transAxes, 
                fontsize=16, fontweight='bold', va='bottom', ha='right')
        if i == 0:
            ax_R.set_xlim(5, 60)
        else:
            ax_R.set_xlim(3, 60)
            
        if i == 0: 
            ax_R.set_ylabel(r"Responsiveness $R_i$", fontsize=12)
        
        # 让 matplotlib 自动寻找最佳位置 (loc='best') 避开数据和 Inset
        ax_R.legend(fontsize=8, frameon=False, loc='upper left')

        # --- 2. Draw Inset Network ---
        ax_ins = ax_R.inset_axes(cfg["inset_bounds"]) 
        draw_network_inset(ax_ins, cfg["type"], cfg["colors"])

        # --- 3. Draw Phase Diagram (b/c) ---
        ax_BC.plot(cfg["N"], cfg["bc"], color='black', lw=2.5, zorder=10)
        
        y_max = cfg["ylim_bc"][1]
        y_min = cfg["ylim_bc"][0]
        ax_BC.set_ylim(y_min, y_max)
        
        if i == 0:
            ax_BC.set_xlim(5, 60)
        else:
            ax_BC.set_xlim(3, 60)
        
        # Fill Regions
        ax_BC.fill_between(cfg["N"], cfg["bc"], 9999, color=c_red, alpha=0.15, label='Cooperation')
        ax_BC.fill_between(cfg["N"], -9999, cfg["bc"], color=c_blue, alpha=0.15, label='Defection')
            
        ax_BC.grid(True, linestyle=':', alpha=0.5)
        ax_BC.set_xlabel(r"Size Parameter $N$", fontsize=12)
        
        # 刻度设置
        if i == 0:
            ax_BC.set_xticks([10, 20, 30, 40, 50, 60])
        else:
            ax_BC.set_xticks([10, 20, 30, 40, 50, 60])
        
        if i == 0:
            ax_BC.set_ylabel(r"Benefit-to-Cost Ratio $b/c$", fontsize=12)
            ax_BC.text(0.65, 0.85, "Cooperation", transform=ax_BC.transAxes, 
                       ha='center', color=c_red, fontsize=11, fontweight='bold')
            ax_BC.text(0.5, 0.15, "Defection", transform=ax_BC.transAxes, 
                       ha='center', color=c_blue, fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(".\\saves\\imgs\\PGFs.pdf", bbox_inches='tight',dpi=500)
    plt.show()

plot_final_with_insets_adjusted()