import numpy as np
import matplotlib.pyplot as plt

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")

width = 432.48195
fig, ax = plt.subplots(
    2,
    1,
    figsize=set_size(width, subplots=(2, 1), aspect_ratio=0.45),
    sharex=True,
    sharey=True,
)
ax[0].plot(
    np.load(
        "runs/price-forecast/DQN_piecewise_linear_18_w1000pd5_2/DQN_piecewise_linear_18_w1000pd5_2_reward.npy"
    ),
    label="DQN piecewise linear",
)
ax[0].plot(
    np.load(
        "runs/price-forecast/DQN_piecewise_quadratic_18_w1000pd5_1/DQN_piecewise_quadratic_18_w1000pd5_1_reward.npy"
    ),
    label="DQN piecewise quadratic",
)
ax[0].plot(
    np.load(
        "runs/price-forecast/PPO_piecewise_linear_18_w1000pc_4/PPO_piecewise_linear_18_w1000pc_4_reward.npy"
    ),
    label="PPO piecewise linear",
)
ax[0].plot(
    np.load(
        "runs/price-forecast/PPO_piecewise_quadratic_18_w1000pc_9/PPO_piecewise_quadratic_18_w1000pc_9_reward.npy"
    ),
    label="PPO piecewise quadratic",
)
ax[0].set_ylabel("mean reward per episode")
ax[0].set_yscale("symlog")
ax[0].legend(loc="lower right")

ax[1].plot(
    np.load(
        "runs/price-forecast/DQN_piecewise_linear_18_w1000pd5_2/DQN_piecewise_linear_18_w1000pd5_2_reward.npy"
    ),
    label="reward",
)
ax[1].plot(
    np.load(
        "runs/price-forecast/DQN_piecewise_linear_18_w1000pd5_2/DQN_piecewise_linear_18_w1000pd5_2_temp_reward.npy"
    ),
    label="temperature reward",
)
ax[1].plot(
    np.load(
        "runs/price-forecast/DQN_piecewise_linear_18_w1000pd5_2/DQN_piecewise_linear_18_w1000pd5_2_price_reward.npy"
    ),
    label="price reward",
)
ax[1].set_ylabel("mean rewards for DQN piecewise linear")
ax[1].set_yscale("symlog")
ax[1].legend(loc="lower right")

# ax[2].plot(
#     np.load(
#         "runs/price-forecast/DQN_piecewise_quadratic_18_w1000pd5_1/DQN_piecewise_quadratic_18_w1000pd5_1_reward.npy"
#     ),
#     label="reward",
# )
# ax[2].plot(
#     np.load(
#         "runs/price-forecast/DQN_piecewise_quadratic_18_w1000pd5_1/DQN_piecewise_quadratic_18_w1000pd5_1_temp_reward.npy"
#     ),
#     label="temperature reward",
# )
# ax[2].plot(
#     np.load(
#         "runs/price-forecast/DQN_piecewise_quadratic_18_w1000pd5_1/DQN_piecewise_quadratic_18_w1000pd5_1_price_reward.npy"
#     ),
#     label="price reward",
# )
# ax[2].set_ylabel("mean rewards for piecewise quadratic")
# ax[2].set_yscale("symlog")
# ax[2].legend(loc="lower right")
# ax[2].set_xlabel("episode")

ax[1].set_xlabel("episode")

fig.savefig("plotting/plots/split_reward_price-forecast.pdf", format="pdf")
