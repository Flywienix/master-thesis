import numpy as np
import matplotlib.pyplot as plt

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")

width = 432.48195
fig, ax = plt.subplots(1, 1, figsize=set_size(width))  # , sharex=True)
ax.plot(
    np.load(
        "runs/method-comparison/DQN_quadratic_18_w1d5_3/DQN_quadratic_18_w1d5_3_reward.npy"
    ),
    label="DQN",
)
ax.plot(
    np.load(
        "runs/method-comparison/PPO_quadratic_18_w1d5_1/PPO_quadratic_18_w1d5_1_reward.npy"
    ),
    label="PPO discrete",
)
ax.plot(
    np.load(
        "runs/method-comparison/PPO_quadratic_18_w1c_4/PPO_quadratic_18_w1c_4_reward.npy"
    ),
    label="PPO continuous",
)
ax.set_ylabel("mean reward (per episode)")
ax.set_yscale("symlog")
ax.legend(loc="lower right")
ax.set_xlabel("episode")
fig.savefig("plotting/plots/reward_method-comparison.pdf", format="pdf")
