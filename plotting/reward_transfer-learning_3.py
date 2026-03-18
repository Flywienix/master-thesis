import numpy as np
import matplotlib.pyplot as plt

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")

width = 432.48195
fig, ax = plt.subplots(1, 1, figsize=set_size(width))  # , sharex=True)
ax.plot(
    np.load(
        "runs/transfer-learning/DQN_piecewise_linear_18_w1000pd5t_3/DQN_piecewise_linear_18_w1000pd5t_3_reward.npy"
    ),
    label="228 to 18, transfer learning",
)
ax.plot(
    np.load(
        "runs/price-forecast/DQN_piecewise_linear_18_w1000pd5_2/DQN_piecewise_linear_18_w1000pd5_2_reward.npy"
    ),
    label="18, normal training",
)
ax.set_ylabel("reward")
ax.set_yscale("symlog")
ax.legend(loc="lower right")
ax.set_xlabel("episode")
fig.savefig("plotting/plots/reward_transfer-learning_3.pdf", format="pdf")
