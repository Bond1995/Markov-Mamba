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
mask = ((idx == 0) + (idx == 2))
sub = idx[mask]
pos = np.where(sub == 2)[0]
pos = pos - np.arange(len(pos))

#
df = []
est = []
for run in api.runs("mamba-markov/markov-mamba-switch"):
    try:
        df.append(run.history(samples=25000))
    except:
        pass

#
for h in df:
    estv = h["est/model_est_0"].values[:]
    estv = estv[~np.isnan(estv)]
    #est = moving_average(est, n=50)
    est.append(estv)
    
est = np.stack(est)
est_mean = np.nanmean(est, axis=0)
est_std = np.nanstd(est, axis=0)

opt_est = df[0]["est/empirical_est_0"].values[:]
opt_est = opt_est[~np.isnan(opt_est)]

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(est_mean, color="tab:green", label="                           ", linewidth=1)
ax.plot(opt_est, color="black", label=" ", linestyle="--", linewidth=1)
ax.fill_between(range(len(est_mean)), est_mean-est_std, est_mean+est_std, color="tab:green", alpha=0.2)
for p in pos:
    plt.axvline(x=p, color="red", linestyle="--", linewidth=0.5)
#ax.set(xlabel="Iterations", ylabel="Test loss")
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
plt.xlim((0,len(est_mean)))
#plt.ylim((0.5,0.7))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(prop={'size': 14}, handlelength=1.7, loc="upper right")
ax.grid(True, which="both")
fig.savefig("switch-estimator.pdf", bbox_inches='tight')