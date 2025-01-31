#%%
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

api = wandb.Api()

def moving_average(a, n=3):
	t = np.floor(n/2).astype(int)
	b = np.zeros(a.shape)
	for i in range(b.shape[-1]):
		b[i] = np.mean(a[max(0, i-t):min(i+t+1, a.shape[-1])])
	
	return b

#
idx = np.load("idx.npy")[0]
pos = np.where(idx == 2)[0]

#
at = []
for i in range(5):
    atv = np.load(f"At_{i}.npy")[0]
    atv = np.exp(atv)
    atv = atv[~np.isnan(atv)]
    at.append(atv)
    
at = np.stack(at)
at_mean = np.nanmean(at, axis=0)
at_std = np.nanstd(at, axis=0)

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(at_mean, color="tab:purple", label="                              ", linewidth=1)
ax.fill_between(range(len(at_mean)), at_mean-at_std, at_mean+at_std, color="tab:purple", alpha=0.2)
for p in pos:
    plt.axvline(x=p, color="red", linestyle="--", linewidth=0.5)
#ax.set(xlabel="Iterations", ylabel="Test loss")
#ax.xaxis.label.set_fontsize(14)
#ax.yaxis.label.set_fontsize(14)
plt.xlim((0,len(at_mean)))
#plt.ylim((0.5,0.7))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
##ax.legend(prop={'size': 14}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("switch-at.pdf", bbox_inches='tight')