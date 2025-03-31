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

df_full = []
loss_full = []
for run in api.runs("mamba-markov/markov-mamba-test-seq"):
    try:
        df_full.append(run.history(samples=25000))
    except:
        pass

df_conv = []
loss_conv = []
for run in api.runs("mamba-markov/markov-mamba-only-conv"):
    try:
        df_conv.append(run.history(samples=25000))
    except:
        pass

df_no_conv = []
loss_no_conv = []
for run in api.runs("mamba-markov/markov-mamba-no-conv"):
    try:
        df_no_conv.append(run.history(samples=25000))
    except:
        pass

#
for h in df_full:
    loss = h["val/loss_gap"].values[:]
    loss = loss[~np.isnan(loss)]
    #est = moving_average(est, n=50)
    loss_full.append(loss)
    
loss_full = np.stack(loss_full)
loss_full_mean = np.nanmean(loss_full, axis=0)
loss_full_std = np.nanstd(loss_full, axis=0)

for h in df_conv:
    loss = h["val/loss_gap"].values[:]
    loss = loss[~np.isnan(loss)]
    #est = moving_average(est, n=50)
    loss_conv.append(loss)
    
loss_conv = np.stack(loss_conv)
loss_conv_mean = np.nanmean(loss_conv, axis=0)
loss_conv_std = np.nanstd(loss_conv, axis=0)

for h in df_no_conv:
    loss = h["val/loss_gap"].values[:]
    loss = loss[~np.isnan(loss)]
    #est = moving_average(est, n=50)
    loss_no_conv.append(loss)
    
loss_no_conv = np.stack(loss_no_conv)
loss_no_conv_mean = np.nanmean(loss_no_conv, axis=0)
loss_no_conv_std = np.nanstd(loss_no_conv, axis=0)

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(loss_full_mean, color="tab:orange", label="                                            ", linewidth=1)
ax.plot(loss_conv_mean, color="tab:green", label=" ", linewidth=1)
ax.plot(loss_no_conv_mean, color="tab:purple", label="                              ", linewidth=1)
ax.fill_between(range(len(loss_full_mean)), loss_full_mean-loss_full_std, loss_full_mean+loss_full_std, color="tab:orange", alpha=0.2)
ax.fill_between(range(len(loss_conv_mean)), loss_conv_mean-loss_conv_std, loss_conv_mean+loss_conv_std, color="tab:green", alpha=0.2)
ax.fill_between(range(len(loss_no_conv_mean)), loss_no_conv_mean-loss_no_conv_std, loss_no_conv_mean+loss_no_conv_std, color="tab:purple", alpha=0.2)
#ax.set(xlabel="Iterations", ylabel="Test loss")
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
plt.xlim((0,39))
plt.ylim((0.0,0.4))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(prop={'size': 14}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("conv-no-conv.pdf", bbox_inches='tight')