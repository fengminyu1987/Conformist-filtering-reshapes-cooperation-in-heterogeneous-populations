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
Ns = {
    "soc-karate": 34,
    "iceland": 75,
    "ca-sandi_auths": 86,
    "contiguous-usa": 49
}

NetLabels = [
    'Iceland',
    'Sandia National Labs',
    'Social Karate Club',
    'USA Contiguous Border'
]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 1.2
})

conformity_types = ['small', 'large']
labels = ["Bottom 10%", "Top 10%"]
cs = ['#E63946', '#3A5A78']

conformist_ratio = 0.1
ssize = 350

# nonlinear conformity strengths, corresponding to file suffixes _2 and _3
kappa_values = [2, 3]

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
# axes[row, col]

for row_idx, kappa in enumerate(kappa_values):
    for col_idx, network_type in enumerate(Gs):

        ax = axes[row_idx, col_idx]
        N = Ns[network_type]

        xmax, ymax, xmin, ymin = 0, 0, 10, 10

        for conformity_type in conformity_types:

            type_idx = conformity_types.index(conformity_type)

            filename = (
                f'./saves/{network_type}_{N}_'
                f'{conformist_ratio:.2f}_{conformity_type}_{kappa}.json'
            )

            with open(filename, 'r') as f:
                data = json.load(f)

            Bf = [float(b) for b in data.keys()]
            rhoc = list(data.values())

            xmax = max(xmax, max(Bf))
            xmin = min(xmin, min(Bf))
            ymax = max(ymax, max(rhoc))
            ymin = min(ymin, min(rhoc))

            THEOREM = (max(Bf) + min(Bf)) / 2

            ax.scatter(
                Bf,
                N * np.array(rhoc),
                s=ssize,
                marker='s',
                facecolors=cs[type_idx],
                edgecolors='white',
                alpha=0.85,
                zorder=3,
                label=labels[type_idx]
            )

            ax.axvline(
                THEOREM,
                ls='--',
                color=cs[type_idx],
                lw=2.5,
                alpha=0.4,
                zorder=1
            )

            z = np.polyfit(Bf, N * np.array(rhoc), 1)
            p = np.poly1d(z)
            x = np.linspace(min(Bf), max(Bf), 100)

            ax.plot(
                x,
                p(x),
                lw=3,
                color=darken(cs[type_idx], 0.7)
            )

        # y = 1 neutral drift line
        ax.axhline(
            1,
            color='purple',
            ls='--',
            lw=2.5,
            alpha=0.6,
            zorder=0
        )

        # cosmetics
        ax.set_xlim(xmin - 0.25, xmax + 0.25)
        ax.margins(y=0.08)

        ax.tick_params(
            axis='both',
            which='major',
            direction='in',
            length=8,
            width=2,
            labelsize=14,
            top=True,
            right=True
        )

        ax.minorticks_on()

        ax.tick_params(
            axis='both',
            which='minor',
            direction='in',
            length=4,
            width=1,
            top=True,
            right=True
        )

        # panel label: a, b, c, ..., h
        panel_idx = row_idx * len(Gs) + col_idx
        ax.text(
            -0.05,
            1.05,
            f"{chr(ord('a') + panel_idx)}",
            transform=ax.transAxes,
            fontsize=22,
            fontweight='bold',
            va='bottom',
            ha='right'
        )

        # network names at the top of each column
        if row_idx == 0:
            ax.set_title(
                NetLabels[col_idx],
                fontsize=17,
                fontfamily="serif",
                pad=16
            )

        # nonlinear conformity strength at the left of each row
        if col_idx == 0:
            ax.text(
                -0.28,
                0.50,
                rf"$\kappa={kappa}$",
                transform=ax.transAxes,
                fontsize=22,
                fontfamily="serif",
                rotation=90,
                va='center',
                ha='center'
            )

# shared labels
fig.text(
    0.5,
    0.04,
    r"$b/c$",
    ha='center',
    fontsize=25,
    fontfamily="serif"
)

fig.text(
    0.05,
    0.5,
    r"$N\times\rho_C$",
    va='center',
    rotation=90,
    fontsize=25,
    fontfamily="serif"
)

# legend only once
axes[0, 0].legend(frameon=False, fontsize=14)

plt.tight_layout(rect=[0.07, 0.06, 1, 0.96])
plt.savefig("./saves/imgs/Fixation Probability Nonlinear Conformity.pdf", dpi=600, bbox_inches="tight")
plt.show()