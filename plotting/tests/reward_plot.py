import numpy as np
import matplotlib.pyplot as plt

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")

width = 432.48195
fig, ax = plt.subplots(1, 1, figsize=set_size(width))  # , sharex=True)
ax.plot(
    np.load(
        "runs/method-comparison/PPO_quadratic_18_w1d5_1/PPO_quadratic_18_w1d5_1_reward.npy"
    ),
    label="reward",
)
ax.set_ylabel("reward")
# ax[0].legend()
ax.set_xlabel("episode")
fig.savefig("plotting/plots/REWARD_TEST.pdf", format="pdf")
