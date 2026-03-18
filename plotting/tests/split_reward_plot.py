import numpy as np
import matplotlib.pyplot as plt

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")

width = 432.48195
fig, ax = plt.subplots(1, 1, figsize=set_size(width))  # , sharex=True)
ax.plot(
    np.load(
        "runs/price-forecast/DQN_piecewise_linear_18_w1000pd5_2/DQN_piecewise_linear_18_w1000pd5_2_reward.npy"
    ),
    label="reward",
)
ax.plot(
    np.load(
        "runs/price-forecast/DQN_piecewise_linear_18_w1000pd5_2/DQN_piecewise_linear_18_w1000pd5_2_temp_reward.npy"
    ),
    label="temp reward",
)
ax.plot(
    np.load(
        "runs/price-forecast/DQN_piecewise_linear_18_w1000pd5_2/DQN_piecewise_linear_18_w1000pd5_2_price_reward.npy"
    ),
    label="price reward",
)
ax.set_ylabel("mean reward of timesteps in episode")
ax.set_yscale("symlog")
ax.legend(loc="lower right")
ax.set_xlabel("episode")
fig.savefig("plotting/plots/REWARD_TEST.pdf", format="pdf")
