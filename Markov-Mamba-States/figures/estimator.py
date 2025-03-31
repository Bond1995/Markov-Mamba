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

df_t1 = []
est_t1 = []
for run in api.runs("mamba-markov/markov-LLM-test-seq", {"config.n_layer": 1}):
    try:
        df_t1.append(run.history(samples=25000))
    except:
        pass

df_t2 = []
est_t2 = []
for run in api.runs("mamba-markov/markov-LLM-test-seq", {"config.n_layer": 2}):
    try:
        df_t2.append(run.history(samples=25000))
    except:
        pass

df_m = []
est_m = []
for run in api.runs("mamba-markov/markov-mamba-test-seq"):
    try:
        df_m.append(run.history(samples=25000))
    except:
        pass

#
for h in df_t1:
    est = h["est/model_est_0"].values[:]
    est = est[~np.isnan(est)]
    #est = moving_average(est, n=50)
    est_t1.append(est)
    
est_t1 = np.stack(est_t1)
est_t1_mean = np.nanmean(est_t1, axis=0)
est_t1_std = np.nanstd(est_t1, axis=0)

for h in df_t2:
    est = h["est/model_est_0"].values[:]
    est = est[~np.isnan(est)]
    #est = moving_average(est, n=50)
    est_t2.append(est)
    
est_t2 = np.stack(est_t2)
est_t2_mean = np.nanmean(est_t2, axis=0)
est_t2_std = np.nanstd(est_t2, axis=0)

for h in df_m:
    est = h["est/model_est_0"].values[:]
    est = est[~np.isnan(est)]
    #est = moving_average(est, n=50)
    est_m.append(est)
    
est_m = np.stack(est_m)
est_m_mean = np.nanmean(est_m, axis=0)
est_m_std = np.nanstd(est_m, axis=0)

opt_est = df_m[0]["est/empirical_est_w0_0"].values[:]
opt_est = opt_est[~np.isnan(opt_est)]

#
sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax.plot(est_m_mean, color="tab:orange", label=" ", linewidth=1)
ax.plot(est_t1_mean, color="tab:purple", label="                              ", linewidth=1)
ax.plot(est_t2_mean, color="tab:green", label=" ", linewidth=1)
ax.plot(opt_est, color="black", label=" ", linestyle="--", linewidth=1)
ax.fill_between(range(len(est_m_mean)), est_m_mean-est_m_std, est_m_mean+est_m_std, color="tab:orange", alpha=0.2)
ax.fill_between(range(len(est_t1_mean)), est_t1_mean-est_t1_std, est_t1_mean+est_t1_std, color="tab:purple", alpha=0.2)
ax.fill_between(range(len(est_t2_mean)), est_t2_mean-est_t2_std, est_t2_mean+est_t2_std, color="tab:green", alpha=0.2)
#ax.set(xlabel="Iterations", ylabel="Test loss")
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
plt.xlim((0,74))
#plt.ylim((0.5,0.7))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(prop={'size': 14}, handlelength=1.7)
ax.grid(True, which="both")
fig.savefig("estimator.pdf", bbox_inches='tight')