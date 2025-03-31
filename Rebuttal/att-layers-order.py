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
orders = np.arange(0, 7, 2)  # the label locations
width = 0.35  # the width of the bars
multiplier = -1.5
fig, ax = plt.subplots()
colors = ["tab:blue", "tab:orange", "tab:green"] #"tab:purple", "tab:red"

# One layer
df = []
loss_mean = []
loss_std = []
for order in range(1, 5):
    losses = []
    for run in api.runs("mamba-markov/markov-LLM-order-deep", {"$and":[{"config.order": order}, {"config.n_layer": 1}]}):
        h = run.history(samples=25000)
        loss = h["val/loss_gap"].values[:]
        loss_min = np.nanmin(loss)
        losses.append(loss_min)
    loss_mean.append(np.mean(losses))
    loss_std.append(np.std(losses))
loss_mean = np.array(loss_mean)
loss_std = np.array(loss_std)

offset = width * multiplier
ax.bar(orders + offset, loss_mean, width, color="tab:red", label="1 layer")
ax.errorbar(orders + offset, loss_mean, yerr=loss_std, fmt='none', ecolor='black', capsize=5)
multiplier += 1

# More layers
for n_layer, color in zip(range(2, 5), colors):
    df = []
    loss_mean = []
    loss_std = []
    for order in range(1, 5):
        losses = []
        for run in api.runs("mamba-markov/markov-LLM-order-deep", {"$and":[{"config.order": order}, {"config.n_layer": n_layer}]}):
            h = run.history(samples=25000)
            loss = h["val/loss_gap"].values[:]
            loss_min = np.nanmin(loss)
            losses.append(loss_min)
        loss_mean.append(np.mean(losses))
        loss_std.append(np.std(losses))
    loss_mean = np.array(loss_mean)
    loss_std = np.array(loss_std)
    
    offset = width * multiplier
    ax.bar(orders + offset, loss_mean, width, color=color, label=str(n_layer)+" layers")
    ax.errorbar(orders + offset, loss_mean, yerr=loss_std, fmt='none', ecolor='black', capsize=5)
    multiplier += 1

    #ax.plot(range(2, 7), loss_mean, color=color, label="           ", linewidth=1)
    #ax.fill_between(range(2, 7), loss_mean-loss_std, loss_mean+loss_std, color=color, alpha=0.2)

ax.set(xlabel="Order", ylabel="Test loss")
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
#plt.xlim((2,6))
plt.ylim((0.0,0.35))
plt.xticks(fontsize=14)
ax.set_xticks(orders, ["1", "2", "3", "4"])
plt.yticks(fontsize=14)
ax.legend(prop={'size': 14}, handlelength=1.7)
ax.grid(True, which="both", axis="y")
ax.grid(False, which="both", axis="x")
fig.savefig("att-layers-order.pdf", bbox_inches='tight')