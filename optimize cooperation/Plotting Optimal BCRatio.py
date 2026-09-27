import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

# --- 1. 基础配置 ---
Graphs = ['petster-hamster-household', 'arenas-email', 'web-polblogs', 'bio-CE-LC', 'power-eris1176']
NetLabels = ["Hamster Owners", "Email", "PolBlogs", "WormNet", "Power Grid"]
data_dir = ".\\saves"
rev = 'asc'

panel_labels = ['a', 'b', 'c', 'd', 'e']

# 定义“高级感”配色方案
styles = {
    # --- 基准线 (最重要) ---
    'Optimal':     {'color': '#D62728', 'ls': '-',      'lw': 2.5, 'alpha': 1.0, 'zorder': 10, 'label': 'Optimal'}, # 深红实线

    # --- 现代物理指标 ---
    'CI':          {'color': '#17BECF', 'ls': (0, (5, 1)),    'lw': 2.0, 'alpha': 0.9, 'zorder': 8,  'label': 'CI'},      # 青色长虚线
    'K-Shell':     {'color': '#E377C2', 'ls': '-.',           'lw': 1.8, 'alpha': 0.8, 'zorder': 7,  'label': 'K-Shell'}, # 粉色点划线

    # --- 经典拓扑指标 ---
    'Degree':      {'color': '#1F77B4', 'ls': '--',           'lw': 1.5, 'alpha': 0.9, 'zorder': 6,  'label': 'Degree'},  # 深蓝虚线
    'Betweenness': {'color': '#2CA02C', 'ls': ':',            'lw': 1.8, 'alpha': 0.8, 'zorder': 5,  'label': 'Betweenness'}, # 翠绿点线
    
    # --- 次要指标 ---
    'Closeness':   {'color': '#FF7F0E', 'ls': (0, (3, 1, 1, 1)), 'lw': 1.5, 'alpha': 0.6, 'zorder': 4,  'label': 'Closeness'}, # 橙色疏点划线
    'Eigenvector': {'color': '#9467BD', 'ls': (0, (5, 5)),    'lw': 1.5, 'alpha': 0.6, 'zorder': 3,  'label': 'Eigenvector'}, # 紫色疏虚线
    'PageRank':    {'color': '#8C564B', 'ls': ':',            'lw': 1.5, 'alpha': 0.6, 'zorder': 2,  'label': 'PageRank'},  # 棕色点线
    
    # --- 随机 Baseline (修改为 Median 版) ---
    'Random_Median': {'color': '#7F7F7F', 'ls': '-',          'lw': 2.0, 'alpha': 0.8, 'zorder': 1,  'label': 'Random'}     # 灰色实线，zorder极低
}

# 设置全局字体
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 1.2, 
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2
})

# --- 2. 绘图逻辑 ---

# 创建 1行5列 的长画布
fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)

y_min_limit = 0.9 

for i, g_name in enumerate(Graphs):
    ax = axes[i]
    label_name = NetLabels[i]
    if rev == 'asc':
        csv_path = os.path.join(data_dir, f"{g_name}_bcr_asc.csv")
    else:
        csv_path = os.path.join(data_dir, f"{g_name}_bcr.csv")
        
    if not os.path.exists(csv_path):
        continue
        
    df = pd.read_csv(csv_path)
    
    # 数据清洗
    cols_to_plot = [c for c in df.columns if c != 'fraction_conformists']
    for col in cols_to_plot:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    x = df['fraction_conformists']
    
    # --- 绘制所有策略曲线 ---
    for strategy, params in styles.items():
        if strategy in df.columns and not df[strategy].isna().all():
            ax.plot(x, df[strategy], **params)

    # --- 绘制 Random Baseline 的透明阴影带 (中位数+分位数) ---
    if 'Random_Median' in df.columns and 'Random_Q_Lower' in df.columns and 'Random_Q_Upper' in df.columns:
        y_lower = df['Random_Q_Lower']
        y_upper = df['Random_Q_Upper']
        
        # 直接使用真实的分位数上下界，无需任何人工截断 (no np.clip!)
        ax.fill_between(x, y_lower, y_upper, 
                        color=styles['Random_Median']['color'], 
                        alpha=0.25, 
                        zorder=0, 
                        edgecolor='none')

    # --- 关键优化点 ---
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter()) 
    
    # 标签放在图内
    if rev == 'asc':
        ax.text(0.05, 0.1, label_name, transform=ax.transAxes, 
            ha='left', va='top', fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    else:
        ax.text(0.05, 0.95, label_name, transform=ax.transAxes, 
            ha='left', va='top', fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

    # 坐标轴标签
    if i == 0:
        ax.set_ylabel(r"Critical Threshold $(b/c)^*$")
    else:
        ax.set_ylabel("") 
        
    ax.set_xticks([0,0.5,1.0])
    ax.set_xlabel(r"Ratio of Conformists")
    
    # 美化网格
    ax.grid(True, which="major", ls="-", color='gray', alpha=0.2, zorder=0.5)
    ax.grid(True, which="minor", ls=":", color='gray', alpha=0.1, zorder=0.5) 

    ax.set_ylim(bottom=y_min_limit)
    ax.text(
        -0.12, 1.1, panel_labels[i],
        transform=ax.transAxes,
        fontsize=20, fontweight='bold',
        va='top', ha='left'
    )

# --- 3. 统一图例 ---
handles, labels = axes[0].get_legend_handles_labels()

# 图例放在最下方
fig.legend(handles, labels, loc='lower center', 
           bbox_to_anchor=(0.5, -0.1), ncol=9, 
           fontsize=16, frameon=False) 

# 创建保存目录
os.makedirs(".\\saves\\imgs", exist_ok=True)

# 保存
if rev == 'asc':
    save_path = ".\\saves\\imgs\\Optimize_BCR_asc(random).pdf"
else:
    save_path = ".\\saves\\imgs\\Optimize_BCR(random).pdf"

plt.savefig(save_path, dpi=500, bbox_inches='tight') 
print(f"Saved to {save_path}")

plt.show()