# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 23:48:38 2026

@author: ziyan
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ==========================================
# 1. 配置参数与风格 (PNAS/Nature Style)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 12

# 定义颜色和标记 (区分度高且美观)
colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F'] # Nature Palettes
#colors = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
markers = ['o', 's', '^', 'D', 'v']
linestyles = ['-', '-', '-', '-', '-']

Graphs = ['petster-hamster-household','arenas-email', 'web-polblogs','bio-CE-LC','power-eris1176']
NetLabels = ["Hamster Owners", "Email", "PolBlogs", "WormNet", "Power Grid"]

bc_stars = {
    'petster-hamster-household': 10.21, 
    'arenas-email': 10.18, 
    'web-polblogs': 8.16, 
    'bio-CE-LC': 2.85,
    'power-eris1176': 12.54
}

# ==========================================
# 2. 数据读取与绘图
# ==========================================
fig, ax = plt.subplots(figsize=(8, 7), dpi=500)

# 加载归一化的 X 轴数据 (Normalized b/c)
xaxis_norm = np.loadtxt("./saves/x_optimal_SCC.txt", delimiter=',')

# --- 主图 (Main Plot): 归一化视角 ---
for i, name in enumerate(Graphs):
    # 加载 Y 轴数据 (Fraction of Negative Nodes)
    try:
        yaxis = np.loadtxt(f"./saves/y_{name}_SCC.txt", delimiter=',')
        
        # 绘制主图曲线
        ax.plot(xaxis_norm, yaxis, 
                color=colors[i], 
                marker=markers[i], 
                markersize=5, 
                markevery=int(len(xaxis_norm)/6), # 防止标记太密
                linewidth=2.3, 
                label=NetLabels[i],
                alpha=0.9)
    except OSError:
        print(f"File not found for {name}")

# --- 装饰主图 ---
ax.axvline(1.0,ls='--',color='0.4',lw=1.5)

ax.text(
    1.0,1.02,r"$x=1$",
    ha='center',va='bottom',
    transform=ax.get_xaxis_transform(),
    fontsize=12,color='0.4'
)

ax.set_xlabel(r"Normalized Benefit-to-Cost Ratio, $x = (b/c) / (b/c)^*_{ori}$")
ax.set_ylabel(r"Required Fraction of Conformists, $\Phi_{neg}$")
ax.set_xlim(0.5, 1.5) # 根据你的数据范围调整
ax.set_ylim(-0.02, 1.02)
ax.legend(frameon=False, loc='lower left',fontsize=16) # 去掉图例边框更显简洁
#ax.axvspan(0.95,1.05,color='0.8',alpha=0.15,zorder=0)
# ==========================================
# 3. 嵌入图 (Inset): 原始视角
# ==========================================
# 在右上角创建一个小图 (位置和大小可微调: width, height, loc)
ax_ins = inset_axes(ax, width="40%", height="35%", loc='upper right', borderpad=2)

for i, name in enumerate(Graphs):
    try:
        yaxis = np.loadtxt(f"./saves/y_{name}_SCC.txt", delimiter=',')
        
        # 还原原始 X 轴: raw_x = normalized_x * bc_star
        raw_xaxis = xaxis_norm * bc_stars[name]
        
        ax_ins.plot(raw_xaxis, yaxis, 
                    color=colors[i], 
                    linewidth=1.5, 
                    alpha=0.8)
    except:
        pass

# --- 装饰嵌入图 ---
ax_ins.set_xlabel(r"$b/c$", fontsize=10)
ax_ins.set_ylabel(r"$\Phi_{neg}$", fontsize=10, labelpad=-2)
ax_ins.tick_params(axis='both', which='major', labelsize=8)
ax_ins.set_facecolor('white')
ax_ins.grid(False)
ax_ins.grid(True, linestyle=':', alpha=0.5)

# ==========================================
# 4. 输出
# ==========================================
plt.tight_layout()
plt.savefig(".\\saves\\imgs\\Optimize_SCC.pdf", bbox_inches='tight', dpi=500)
plt.show()