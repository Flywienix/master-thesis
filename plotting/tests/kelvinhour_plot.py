import os

import numpy as np
import matplotlib.pyplot as plt

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")

intermediate_model_path = (
    "runs/method-comparison/PPO_quadratic_18_w1d5_1/PPO_quadratic_18_w1d5_1_model_"
)
path_parts = intermediate_model_path.split("/")
model_folder = "/".join(path_parts[:-1])
intermediate_evaluation_folder = model_folder + "/intermediate_evaluation"
intermediate_temps = list()
for i in range(1, 30 + 1):
    # os.makedirs(intermediate_evaluation_folder + "/" + str(i * 100), exist_ok=True)
    # os.system(
    #     "python evaluate.py "
    #     + intermediate_model_path
    #     + str(i * 100)
    #     + " 18 -o "
    #     + intermediate_evaluation_folder
    #     + "/"
    #     + str(i * 100)
    # )
    temp = np.load(
        intermediate_evaluation_folder
        + "/"
        + str(i * 100)
        + "/"
        + path_parts[-2]
        + "_temp.npy"
    )
    intermediate_temps.append((i * 100, temp))
with open(model_folder + "/info.txt", "r") as info_file:
    best_episode = int(info_file.readlines()[-1].split(" ")[-1])
intermediate_temps.insert(
    int(best_episode / 100),
    (best_episode, np.load(intermediate_model_path[:-6] + "temp.npy")),
)
kelvinhours = [(i, np.mean(np.abs(x - 21.0) * 0.25)) for (i, x) in intermediate_temps]
print(intermediate_temps)

width = 432.48195
fig, ax = plt.subplots(1, 1, figsize=set_size(width))  # , sharex=True)
ax.plot(
    [i for (i, x) in kelvinhours],
    [x for (i, x) in kelvinhours],
    label="kelvinhours",
)
ax.set_ylabel("mean kelvinhours per episode")
# ax[0].legend()
ax.set_xlabel("episode")
fig.savefig("plotting/plots/KELVINHOURS_TEST.pdf", format="pdf")
