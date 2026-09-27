# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 01:32:53 2026

@author: ziyan
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.colors import to_rgb

def darken(color, factor=0.6):
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)

plt.rcParams['axes.linewidth'] = 2

Gs = ["iceland", "ca-sandi_auths", "soc-karate", "contiguous-usa"]
Ns = {"soc-karate": 34, "iceland": 75, "ca-sandi_auths": 86, "contiguous-usa": 49}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 1.2
})

# 行对应的从众者类型：第一行 large(top 10%), 第二行 small(bottom 10%)
conformity_types = ['large', 'small']
row_labels = ['Top 10%', 'Bottom 10%']

# 三个选择强度
selection_strengths = [0.10, 1.00]
strength_labels = [r'$\delta=0.10$', r'$\delta=1.00$']

# 你也可以换成你喜欢的颜色
cs = ['#E63946', '#F4A261', '#3A5A78']

conformist_ratio = 0.1
ssize = 300
NetLabels = ['Iceland', 'Sandia National Labs', 'Social Karate Club', 'USA Contiguous Border']

fig, axes = plt.subplots(2, 4, figsize=(22, 10))

for row_idx, conformity_type in enumerate(conformity_types):
    for col_idx, network_type in enumerate(Gs):

        ax = axes[row_idx, col_idx]
        N = Ns[network_type]

        xmax, ymax, xmin, ymin = 0, 0, 10, 10

        # 画三个选择强度
        for s_idx, sel in enumerate(selection_strengths):

            sel_str = float(f"{sel:.2f}")   # 0.1 -> "0.1", 1 -> "1", 10 -> "10"
            filename = f'./saves/{network_type}_{N}_{conformist_ratio:.2f}_{conformity_type}_{sel_str:.2f}.json'
            data = json.load(open(filename))

            Bf = [float(b) for b in data.keys()]
            rhoc = list(data.values())

            xmax = max(xmax, max(Bf))
            xmin = min(xmin, min(Bf))
            ymax = max(ymax, max(rhoc))
            ymin = min(ymin, min(rhoc))

            ax.scatter(
                Bf,
                N * np.array(rhoc),
                s=ssize,
                marker='s',
                facecolors=cs[s_idx],
                edgecolors='white',
                alpha=0.85,
                zorder=3,
                label=strength_labels[s_idx]
            )

        # 中性漂移基线 y=1
        ax.axhline(1, color='purple', ls='--', lw=2.5, alpha=0.6, zorder=0)

        # cosmetics
        ax.set_xlim(xmin - 0.25, xmax + 0.25)
        ax.margins(y=0.08)

        ax.tick_params(
            axis='both', which='major', direction='in',
            length=8, width=2, labelsize=14, top=True, right=True
        )
        ax.minorticks_on()
        ax.tick_params(
            axis='both', which='minor', direction='in',
            length=4, width=1, top=True, right=True
        )

        # panel label
        panel_id = row_idx * 4 + col_idx
        ax.text(
            -0.05, 1.05, f"{chr(ord('a') + panel_id)}",
            transform=ax.transAxes,
            fontsize=22, fontweight='bold',
            va='bottom', ha='right'
        )

        # 第一行上方写网络名
        if row_idx == 0:
            ax.text(
                0.03, 1.08, NetLabels[col_idx],
                transform=ax.transAxes,
                fontsize=16,
                fontfamily="serif",
                va='bottom',
                ha='left'
            )

        # 每一行左边写 Top/Bottom 10%
        if col_idx == 0:
            ax.text(
                -0.12, 0.50, row_labels[row_idx],
                transform=ax.transAxes,
                fontsize=18,
                fontfamily="serif",
                rotation=90,
                va='center',
                ha='center'
            )

# shared labels
fig.text(0.5, 0.04, r"$b/c$", ha='center', fontsize=25, fontfamily="serif")
fig.text(0.03, 0.5, r"$N\times\rho_C$", va='center', rotation=90, fontsize=25, fontfamily="serif")

# legend only once
axes[0, 0].legend(frameon=False, fontsize=13)

plt.tight_layout(rect=[0.06, 0.06, 1, 1])
plt.savefig("./saves/imgs/Fixation Probability Stronger Selection.pdf", dpi=600)
plt.show()