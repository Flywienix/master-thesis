import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")
color_palette = sns.color_palette()
print(color_palette)

width = 432.48195

price = np.load("data/price_val.npy")
start, end = 0, price.shape[0]
print(end)
# start = 1000
# end = start + 5 * 96
fig, ax = plt.subplots(
    1, 1, figsize=set_size(width, aspect_ratio=0.3)
)  # , sharex=True)
ax.plot(price[start:end], label="price data", color=color_palette[1])
ax.set_ylabel("Price data\n")
ax.set_xlabel("Time in 15 min steps (i.e. 96 = 24 hours)")
fig.savefig("plotting/plots/price_val_data.pdf", format="pdf")
