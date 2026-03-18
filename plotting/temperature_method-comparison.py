import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")
color_palette = sns.color_palette()

width = 432.48195

temperature_DQN_discrete = np.load(
    "runs/method-comparison/DQN_quadratic_18_w1d5_3/DQN_quadratic_18_w1d5_3_temp.npy"
)
temperature_PPO_discrete = np.load(
    "runs/method-comparison/PPO_quadratic_18_w1d5_1/PPO_quadratic_18_w1d5_1_temp.npy"
)
temperature_PPO_continuous = np.load(
    "runs/method-comparison/PPO_quadratic_18_w1c_4/PPO_quadratic_18_w1c_4_temp.npy"
)
Tamb = np.load("data/Tamb_Csb.npy")

start = 0
duration = temperature_DQN_discrete.shape[0]
fig, ax = plt.subplots(4, 1, figsize=set_size(width, subplots=(4, 1), aspect_ratio=0.37), sharex=True)
ax[0].plot(temperature_DQN_discrete[start : start + duration])
ax[0].plot([21] * duration, alpha=0.5, color=color_palette[2])
ax[0].set_ylabel("Indoor air temperature in °C\nfor discrete DQN")
ax[0].set_ylim([19.9,23.4])
ax[1].plot(temperature_PPO_discrete[start : start + duration])
ax[1].plot([21] * duration, alpha=0.5, color=color_palette[2])
ax[1].set_ylabel("Indoor air temperature in °C\nfor discrete PPO")
ax[1].set_ylim([19.9,23.4])
ax[2].plot(temperature_PPO_continuous[start : start + duration])
ax[2].plot([21] * duration, alpha=0.5, color=color_palette[2])
ax[2].set_ylabel("Indoor air temperature in °C\nfor continuous PPO")
ax[2].set_ylim([19.9,23.4])
ax[3].plot(
    Tamb[start : start + duration],
    label="outdoor temperature",
)
ax[3].set_ylabel("Ambient air temperature in °C\n")
ax[3].set_xlabel("Time in 15 min steps")
fig.savefig("plotting/plots/temperature_method-comparison.pdf", format="pdf")
