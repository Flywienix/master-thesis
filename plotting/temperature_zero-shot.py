import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")
color_palette = sns.color_palette()

width = 432.48195

zero_shot_16 = np.load(
    "runs/transfer-learning/zero-shot/DQN_18_to_16_zs/DQN_piecewise_linear_18_w1000pd5_2_temp.npy"
)
zero_shot_63 = np.load(
    "runs/transfer-learning/zero-shot/DQN_18_to_63_zs/DQN_piecewise_linear_18_w1000pd5_2_temp.npy"
)
zero_shot_228 = np.load(
    "runs/transfer-learning/zero-shot/DQN_18_to_228_zs/DQN_piecewise_linear_18_w1000pd5_2_temp.npy"
)
zero_shot_1404 = np.load(
    "runs/transfer-learning/zero-shot/DQN_18_to_1404_zs/DQN_piecewise_linear_18_w1000pd5_2_temp.npy"
)
zero_shot_1448 = np.load(
    "runs/transfer-learning/zero-shot/DQN_18_to_1448_zs/DQN_piecewise_linear_18_w1000pd5_2_temp.npy"
)
Tamb = np.load("data/Tamb_Csb.npy")

start = 0
duration = zero_shot_228.shape[0]
fig, ax = plt.subplots(
    1, 1, figsize=set_size(width, subplots=(1, 1), aspect_ratio=0.5), sharex=True
)
ax.plot(
    zero_shot_1404[start : start + duration],
    label="0.64994673",
)
ax.plot(
    zero_shot_1448[start : start + duration],
    zorder=10,
    label="0.355545557",
)
# ax.plot(
#     zero_shot_16[start : start + duration],
#     label="17 invalid (index 16 invalid) 0.173424174",
# )
ax.plot(
    zero_shot_228[start : start + duration],
    color=color_palette[3],
    label="0.175385237",
)
ax.plot(
    zero_shot_63[start : start + duration],
    color=color_palette[4],
    label="0.148505814",
)
ax.plot([21] * duration, color=color_palette[2], linestyle="dashed")
ax.set_ylabel("Indoor air temperature in °C")
ax.legend()
ax.set_xlabel("Time in 15 min steps")
fig.savefig("plotting/plots/temperature_zero-shot.pdf", format="pdf")
