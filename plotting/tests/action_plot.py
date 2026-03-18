import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")
color_palette = sns.color_palette()

width = 432.48195

temperature = np.load(
    "runs/method-comparison/PPO_quadratic_18_w1d5_1/PPO_quadratic_18_w1d5_1_temp.npy"
)
# price = np.load(
#     "runs/method-comparison/PPO_quadratic_18_w1d5_1/PPO_quadratic_18_w1d5_1_price.npy"
# )
Tamb = np.load("data/Tamb_Csb.npy")
actions = np.load(
    "runs/method-comparison/PPO_quadratic_18_w1d5_1/PPO_quadratic_18_w1d5_1_actions.npy"
)

start = 0
duration = 7 * 96
fig, ax = plt.subplots(2, 1, figsize=set_size(width, subplots=(2, 1)), sharex=True)
ax[0].plot(temperature[start : start + duration])
ax[0].set_ylabel("Indoor air temperature in °C")
# ax[1].plot(price[start : start + duration], color=color_palette[1], label="price")
ax[1].plot(
    Tamb[start : start + duration],
    label="outdoor temperature",
)
ax2 = ax[1].twinx()
ax2.plot(
    actions[start : start + duration],
    # ".", # dots or connected line?
    color=color_palette[3],
    label="action",
)
ax2.set_ylabel("heatpump activation")
# ax[1].set_ylabel("Price data")
ax[1].set_ylabel("Ambient air temperature in °C")
ax[1].set_xlabel("Time in 15 min steps")
fig.savefig("plotting/plots/ACTION_TEST.pdf", format="pdf")
