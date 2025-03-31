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

df_m = []
loss_m = []
for run in api.runs("mamba-markov/markov-mamba-test-seq"):
    try:
        df_m.append(run.history(samples=25000))
    except:
        pass

df_m1 = []
loss_m1 = []
for run in api.runs("mamba-markov/markov-mamba-no-conv-l1-new"):
    try:
        df_m1.append(run.history(samples=25000))
    except:
        pass

df_mnc = []
loss_mnc = []
for run in api.runs("mamba-markov/markov-mamba-no-conv-test-seq"):
    try:
        df_mnc.append(run.history(samples=25000))
    except:
        pass

#
for h in df_m:
    loss = h["val/loss_gap"].values[:]
    loss = loss[~np.isnan(loss)]
    #est = moving_average(est, n=50)
    loss_m.append(loss)
    
loss_m = np.stack(loss_m)
loss_m_mean = np.nanmean(loss_m, axis=0)
loss_m_std = np.nanstd(loss_m, axis=0)

for h in df_m1:
    loss = h["val/loss_gap"].values[:]
    loss = loss[~np.isnan(loss)]
    #est = moving_average(est, n=50)
    loss_m1.append(loss)
    
loss_m1 = np.stack(loss_m1)
loss_m1_mean = np.nanmean(loss_m1, axis=0)
loss_m1_std = np.nanstd(loss_m1, axis=0)

for h in df_mnc:
    loss = h["val/loss_gap"].values[:]
    loss = loss[~np.isnan(loss)]
    #est = moving_average(est, n=50)
    loss_mnc.append(loss)
    
loss_mnc = np.stack(loss_mnc)
loss_mnc_mean = np.nanmean(loss_mnc, axis=0)
loss_mnc_std = np.nanstd(loss_mnc, axis=0)

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(loss_m_mean, color="tab:orange", label="1-layer Mamba", linewidth=1)
ax.plot(loss_m1_mean, color="tab:purple", label="1-layer Mamba without convolution", linewidth=1)
ax.plot(loss_mnc_mean, color="tab:blue", label="2-layer Mamba without convolution", linewidth=1)
ax.fill_between(range(len(loss_m_mean)), loss_m_mean-loss_m_std, loss_m_mean+loss_m_std, color="tab:orange", alpha=0.2)
ax.fill_between(range(len(loss_m1_mean)), loss_m1_mean-loss_m1_std, loss_m1_mean+loss_m1_std, color="tab:purple", alpha=0.2)
ax.fill_between(range(len(loss_mnc_mean)), loss_mnc_mean-loss_mnc_std, loss_mnc_mean+loss_mnc_std, color="tab:blue", alpha=0.2)
ax.set(xlabel="Iteration (x 200)", ylabel="Test loss")
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
plt.xlim((0,15))
plt.ylim((0.0,0.5))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(prop={'size': 14}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("mamba-no-conv-loss.pdf", bbox_inches='tight')