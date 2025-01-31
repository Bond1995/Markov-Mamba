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
sns.set_style("whitegrid")
orders = np.arange(4)  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots()

# 1-layer Mamba
norm_mean = []
norm_std = []
for order in range(1, 5):
    norms = []
    for run in api.runs("mamba-markov/markov-mamba-conv-order", {"$and":[{"config.order": order}, {"$or":[{"config.d_conv": 5}, {"config.d_conv": 6}]}]}):
        h = run.history(samples=25000)
        norm = 0
        length = 0
        for i in range(2**order):
            est = h["est/model_est_"+str(i)].values[:]
            est = est[~np.isnan(est)]
            opt = h["est/empirical_est_w0_"+str(i)].values[:]
            opt = opt[~np.isnan(opt)]
            norm += np.linalg.norm(est - opt, ord=1)
            length += len(est)
        norm = norm / length
        norms.append(norm)
    norm_mean.append(np.mean(norms))
    norm_std.append(np.std(norms))
norm_mean = np.array(norm_mean)
norm_std = np.array(norm_std)

offset = width * multiplier
ax.bar(orders + offset, norm_mean, width, color="tab:orange", label="               ")
ax.errorbar(orders + offset, norm_mean, yerr=norm_std, fmt='none', ecolor='black', capsize=5)
multiplier += 1

#ax.plot(range(1, 5), norm_mean, color="tab:orange", label="                              ", linewidth=1)
#ax.fill_between(range(1, 5), norm_mean-norm_std, norm_mean+norm_std, color="tab:orange", alpha=0.2)

# 1-layer Transformer
norm_mean = []
norm_std = []
for order in range(1, 5):
    norms = []
    for run in api.runs("mamba-markov/markov-LLM-order-l1", {"config.order": order}):
        h = run.history(samples=25000)
        norm = 0
        length = 0
        for i in range(2**order):
            try:
                est = h["est/model_est_"+str(i)].values[:]
                est = est[~np.isnan(est)]
                opt = h["est/empirical_est_"+str(i)].values[:]
                opt = opt[~np.isnan(opt)]
                norm += np.linalg.norm(est - opt, ord=1)
                length += len(est)
            except:
                pass
        norm = norm / length
        norms.append(norm)
    norm_mean.append(np.mean(norms))
    norm_std.append(np.std(norms))
norm_mean = np.array(norm_mean)
norm_std = np.array(norm_std)

offset = width * multiplier
ax.bar(orders + offset, norm_mean, width, color="tab:purple", label="                              ")
ax.errorbar(orders + offset, norm_mean, yerr=norm_std, fmt='none', ecolor='black', capsize=5)
multiplier += 1

#ax.plot(range(1, 5), norm_mean, color="tab:purple", label="                              ", linewidth=1)
#ax.fill_between(range(1, 5), norm_mean-norm_std, norm_mean+norm_std, color="tab:purple", alpha=0.2)

# 2-layer Transformer
norm_mean = []
norm_std = []
for order in range(1, 5):
    norms = []
    for run in api.runs("mamba-markov/markov-LLM-order", {"config.order": order}):
        h = run.history(samples=25000)
        norm = 0
        length = 0
        for i in range(2**order):
            try:
                est = h["est/model_est_"+str(i)].values[:]
                est = est[~np.isnan(est)]
                opt = h["est/empirical_est_"+str(i)].values[:]
                opt = opt[~np.isnan(opt)]
                norm += np.linalg.norm(est - opt, ord=1)
                length += len(est)
            except:
                pass
        norm = norm / length
        norms.append(norm)
    norm_mean.append(np.mean(norms))
    norm_std.append(np.std(norms))
norm_mean = np.array(norm_mean)
norm_std = np.array(norm_std)

offset = width * multiplier
ax.bar(orders + offset, norm_mean, width, color="tab:green", label="               ")
ax.errorbar(orders + offset, norm_mean, yerr=norm_std, fmt='none', ecolor='black', capsize=5)
multiplier += 1

#ax.plot(range(1, 5), norm_mean, color="tab:green", label="                              ", linewidth=1)
#ax.fill_between(range(1, 5), norm_mean-norm_std, norm_mean+norm_std, color="tab:green", alpha=0.2)

#
#ax.set(xlabel="Iterations", ylabel="Test loss")
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
#plt.xlim((1,4))
plt.ylim((0.0,0.35))
plt.xticks(fontsize=14)
ax.set_xticks(orders + width, ["1", "2", "3", "4"])
plt.yticks(fontsize=14)
ax.legend(prop={'size': 14}, handlelength=1.7)
ax.grid(True, which="both", axis="y")
ax.grid(False, which="both", axis="x")
fig.savefig("estimator-norm.pdf", bbox_inches='tight')