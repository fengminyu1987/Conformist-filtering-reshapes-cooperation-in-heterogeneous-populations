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
    r,g,b = to_rgb(color)
    return (r*factor, g*factor, b*factor)

plt.rcParams['axes.linewidth'] = 2

Gs=["iceland","ca-sandi_auths","soc-karate","contiguous-usa"]
Ns={"soc-karate":34,"iceland":75,"ca-sandi_auths":86,"contiguous-usa":49}
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 1.2
})
conformity_types=['small','large']
labels=["Bottom 10%","Top 10%"]
cs=['#E63946','#3A5A78']
conformist_ratio=0.1
ssize=350
NetLabels = ['Iceland','Sandia National Labs', 'Social Karate Club','USA Contiguous Border']
fig,axes=plt.subplots(2,2,figsize=(12,10))
axes=axes.flatten()

for idx,network_type in enumerate(Gs):

    ax=axes[idx]
    N=Ns[network_type]

    xmax,ymax,xmin,ymin=0,0,10,10
    for conformity_type in conformity_types:

        filename=f'./saves/{network_type}_{N}_{conformist_ratio:.2f}_{conformity_type}.json'
        data=json.load(open(filename))

        Bf=[float(b) for b in data.keys()]
        rhoc=list(data.values())

        xmax=max(xmax,max(Bf)); xmin=min(xmin,min(Bf))
        ymax=max(ymax,max(rhoc)); ymin=min(ymin,min(rhoc))

        THEOREM=(max(Bf)+min(Bf))/2

        ax.scatter(
            Bf,N*np.array(rhoc),
            s=ssize,
            marker='s',
            facecolors=cs[conformity_types.index(conformity_type)],
            edgecolors='white',
            alpha=0.85,zorder=3,
            label=labels[conformity_types.index(conformity_type)]
        )

        ax.axvline(THEOREM,ls='--',color=cs[conformity_types.index(conformity_type)],
                   lw=2.5,alpha=0.4,zorder=1)

        z=np.polyfit(Bf,N*np.array(rhoc),1)
        p=np.poly1d(z)
        x=np.linspace(min(Bf),max(Bf),100)
        ax.plot(x,p(x),lw=3,color=darken(cs[conformity_types.index(conformity_type)],0.7))
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    # baseline
    base=json.load(open(f'./saves/{network_type}_{N}.json'))
    Bf=[float(b) for b in base.keys()]
    rhoc=list(base.values())
    THEOREM=(max(Bf)+min(Bf))/2

    ax.scatter(Bf,N*np.array(rhoc),s=ssize,marker='s',
               facecolors="0.6",edgecolors='white',alpha=0.85,zorder=3,
               label="No Conformists")

    z=np.polyfit(Bf,N*np.array(rhoc),1)
    p=np.poly1d(z)
    x=np.linspace(min(Bf),max(Bf),100)
    ax.plot(x,p(x),lw=3,color="black")

    ax.axvline(THEOREM,ls='--',color="0.5",lw=2,alpha=0.4)

    # y=1
    ax.axhline(1,color='purple',ls='--',lw=2.5,alpha=0.6,zorder=0)

    # cosmetics
    ax.set_xlim(xmin-0.25,xmax+0.25)
    ax.margins(y=0.08)

    ax.tick_params(axis='both',which='major',direction='in',
                   length=8,width=2,labelsize=14,top=True,right=True)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='minor',direction='in',
                   length=4,width=1,top=True,right=True)

    # panel label
    ax.text(-0.05, 1.05,f"{chr(ord('a')+idx)}",
            transform=ax.transAxes,fontsize=22,fontweight='bold',va='bottom', ha='right')
    ax.text(
    0.03, 0.95, NetLabels[Gs.index(network_type)],
    transform=ax.transAxes,
    fontsize=16,
    fontfamily="serif",
    va='top',
    ha='left'
)

    #ax.set_title(network_type)

# shared labels
fig.text(0.5,0.04,r"$b/c$",ha='center',fontsize=25,fontfamily="serif")
fig.text(0.04,0.5,r"$N\times\rho_C$",va='center',rotation=90,fontsize=25,fontfamily="serif")

# legend only once
axes[0].legend(frameon=False,fontsize=14)

plt.tight_layout(rect=[0.06,0.06,1,1])
plt.savefig("./saves/imgs/Fixation Probability.pdf",dpi=600)
plt.show()