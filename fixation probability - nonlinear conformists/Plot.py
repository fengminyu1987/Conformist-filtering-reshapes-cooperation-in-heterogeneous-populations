# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:32:26 2024

@author: zziya
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.colors import to_rgb
def darken(color, factor=0.6):
    r,g,b = to_rgb(color)
    return (r*factor, g*factor, b*factor)
plt.rcParams['axes.linewidth'] = 3
Gs=["iceland", "ca-sandi_auths", "soc-karate", "contiguous-usa"]
network_type='contiguous-usa'
Ns={"soc-karate":34,"iceland":75,"ca-sandi_auths":86,"contiguous-usa":49}
N=Ns[network_type]
conformity_type='small'
conformity_types=['small', 'large']
conformist_ratio=0.1
cs=['#E63946', '#3A5A78']#'#F4D35E',
labels=["Top 10%","Bottom 10%"]
ssize=450
plt.figure(figsize=(8,8))
xmax,ymax,xmin,ymin=0,0,10,10
for conformity_type in conformity_types:
    filename='.\\saves\\{}_{:d}_{:.2f}_{}.json'.format(network_type,N,conformist_ratio,conformity_type)
    f=open(filename)
    data=json.load(f)
    B=list(data.keys())
    Bf=[round(float(B[i]),5) for i in range(len(B))]
    if max(Bf)>xmax:
        xmax=max(Bf)
    if min(Bf)<xmin:
        xmin=min(Bf)
    THEOREM=(max(Bf)+min(Bf))/2
    rhoc=list(data.values())
    if max(rhoc)>ymax:
        ymax=max(rhoc)
    if min(rhoc)<ymin:
        ymin=min(rhoc)
    plt.scatter(Bf, N*np.array(rhoc),s=ssize,marker='s',facecolors=cs[conformity_types.index(conformity_type)], edgecolors='white',label=labels[conformity_types.index(conformity_type)],alpha=0.85,zorder=3)
    plt.axvline(
    THEOREM,
    linestyle='--',
    color=cs[conformity_types.index(conformity_type)],
    lw=2.5,
    alpha=0.4,
    zorder=1
)
    z1=np.polyfit(Bf, N*np.array(rhoc), 1)
    p1=np.poly1d(z1)
    plt.plot(np.linspace(min(Bf),max(Bf),100), p1(np.linspace(min(Bf),max(Bf),100)), lw=3, color=darken(cs[conformity_types.index(conformity_type)], 0.7),zorder=2)
#baseline without conformists
filename_baseline='.\\saves\\{}_{:d}.json'.format(network_type,N)
f=open(filename_baseline)
data=json.load(f)
B=list(data.keys())
Bf=[round(float(B[i]),5) for i in range(len(B))]
if max(Bf)>xmax:
    xmax=max(Bf)
if min(Bf)<xmin:
    xmin=min(Bf)
THEOREM=(max(Bf)+min(Bf))/2
plt.scatter(Bf, N*np.array(rhoc),s=ssize,marker='s',facecolors="0.6", edgecolors='white',label="No Conformists",alpha=0.85,zorder=3)
plt.axvline(THEOREM, linestyle='--', color='0.5', lw=2, alpha=0.4)
z1=np.polyfit(Bf, N*np.array(rhoc), 1)
p1=np.poly1d(z1)
plt.plot(np.linspace(min(Bf),max(Bf),100), p1(np.linspace(min(Bf),max(Bf),100)), lw=3, color="black",zorder=2)
#modifications
plt.axhline(1, color='purple', linestyle='--', lw=2.5, alpha=0.6, zorder=0)
plt.xlabel('$b/c$',fontsize=25)
plt.ylabel('$N \\times \\rho_C$',fontsize=25)
plt.margins(x=0.05, y=0.08)
plt.tick_params(
    axis='both',
    which='major',
    direction='in',
    length=8,
    width=2,
    labelsize=20,
    pad=6,
    top=True,
    right=True
)
plt.tick_params(labeltop=False, labelright=False)
plt.minorticks_on()
plt.tick_params(
    axis='both',
    which='minor',
    direction='in',
    length=4,
    width=1,
    top=True,
    right=True
)
plt.legend(frameon=False,loc="best", fontsize=20)
plt.grid(alpha=0.15, linestyle=':', linewidth=1)
plt.savefig('.\\saves\\imgs\\{}.pdf'.format(network_type),dpi=500,bbox_inches='tight')
plt.show()