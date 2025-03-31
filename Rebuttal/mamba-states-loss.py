#%%
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

api = wandb.Api()
sns.set_style("whitegrid")
fig, ax = plt.subplots()

def moving_average(a, n=3):
	t = np.floor(n/2).astype(int)
	b = np.zeros(a.shape)
	for i in range(b.shape[-1]):
		b[i] = np.mean(a[max(0, i-t):min(i+t+1, a.shape[-1])])
	
	return b

for vocab_size, color in zip([2, 4, 6, 8], ["tab:blue", "tab:orange", "tab:green", "tab:purple"]):
    df = []
    losses = []
    for run in api.runs("mamba-markov/markov-mamba-states", {"config.vocab_size": vocab_size}):
        try:
            df.append(run.history(samples=25000))
        except:
            pass

    for h in df:
        loss = h["val/loss_gap"].values[:]
        loss = loss[~np.isnan(loss)]
        #est = moving_average(est, n=50)
        losses.append(loss)
    
    losses = np.stack(losses)
    loss_mean = np.nanmean(losses, axis=0)
    loss_std = np.nanstd(losses, axis=0)

    ax.plot(loss_mean, color=color, label=str(vocab_size)+" states", linewidth=1)
    ax.fill_between(range(len(loss_mean)), loss_mean-loss_std, loss_mean+loss_std, color=color, alpha=0.2)

ax.set(xlabel="Iteration (x 200)", ylabel="Test loss gap")
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
plt.xlim((0,39))
plt.ylim((0.0,0.5))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(prop={'size': 14}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("mamba-states-loss.pdf", bbox_inches='tight')