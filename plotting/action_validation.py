import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")
color_palette = sns.color_palette()

width = 432.48195

temperature = np.load(
    "runs/price-forecast/DQN_piecewise_linear_18_w1000pd5_2/DQN_piecewise_linear_18_w1000pd5_2_temp_val.npy"
)
price = np.load(
    "runs/price-forecast/DQN_piecewise_linear_18_w1000pd5_2/DQN_piecewise_linear_18_w1000pd5_2_price_val.npy"
)
# Tamb = np.load("data/Tamb_Csb.npy")
actions = np.load(
    "runs/price-forecast/DQN_piecewise_linear_18_w1000pd5_2/DQN_piecewise_linear_18_w1000pd5_2_actions_val.npy"
)

start = 8 * 96
duration = 3 * 96
fig, ax = plt.subplots(2, 1, figsize=set_size(width, subplots=(2, 1)), sharex=True)
temperature_line = ax[0].plot(temperature[start : start + duration], label="DQN")
ax[0].fill_between(
    range(duration), [20] * duration, [22] * duration, alpha=0.1, color=color_palette[2]
)
ax[0].set_ylabel("Indoor air temperature in °C")
price_line = ax[1].plot(
    price[start : start + duration], color=color_palette[1], label="price"
)
# outdoor_temp_line = ax[1].plot(
#     Tamb[start : start + duration],
#     label="outdoor temperature",
# )
ax2 = ax[1].twinx()
action_line = ax2.plot(
    actions[start : start + duration],
    # ".", # dots or connected line?
    color=color_palette[3],
    label="action DQN",
)
ax2.set_ylabel("heatpump activation (action)")
ax[1].set_ylabel("Price data")
# ax[1].set_ylabel("Ambient air temperature in °C")
ax[1].set_xlabel("Time in 15 min steps")
fig.savefig("plotting/plots/action_validation.pdf", format="pdf")
