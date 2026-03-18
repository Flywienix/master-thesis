import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")
color_palette = sns.color_palette()

intermediate_model_path_PPO_no_price_forecast = (
    "runs/method-comparison/PPO_quadratic_18_w1c_4/PPO_quadratic_18_w1c_4_model_"
)
intermediate_model_path_DQN_piecewise_linear = "runs/price-forecast/DQN_piecewise_linear_18_w1000pd5_2/DQN_piecewise_linear_18_w1000pd5_2_model_"
intermediate_model_path_DQN_piecewise_quadratic = "runs/price-forecast/DQN_piecewise_quadratic_18_w1000pd5_1/DQN_piecewise_quadratic_18_w1000pd5_1_model_"
intermediate_model_path_PPO_piecewise_linear = "runs/price-forecast/PPO_piecewise_linear_18_w1000pc_4/PPO_piecewise_linear_18_w1000pc_4_model_"
intermediate_model_path_PPO_piecewise_quadratic = "runs/price-forecast/PPO_piecewise_quadratic_18_w1000pc_9/PPO_piecewise_quadratic_18_w1000pc_9_model_"


def calculate_kelvinhours(
    path: str,
    continuous: bool = False,
    price_forecast: bool = True,
    calculate_intermediate: bool = False,
) -> list:
    path_parts = path.split("/")
    model_folder = "/".join(path_parts[:-1])
    intermediate_evaluation_folder = model_folder + "/intermediate_evaluation"
    intermediate_temps = list()
    for i in range(1, 30 + 1):
        if calculate_intermediate:
            os.makedirs(
                intermediate_evaluation_folder + "/" + str(i * 100), exist_ok=True
            )
            call = (
                "python evaluate.py "
                + path
                + str(i * 100)
                + " 18 "
                + "-o "
                + intermediate_evaluation_folder
                + "/"
                + str(i * 100)
                + (" -c" if continuous else "")
                + (" -p" if price_forecast else "")
            )
            os.system(call)
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
        (best_episode, np.load(path[:-6] + "temp.npy")),
    )
    kelvinhours = [
        (i, np.mean(np.abs(x - 21.0) * 0.25)) for (i, x) in intermediate_temps
    ]
    print(kelvinhours, "\n")
    return kelvinhours, best_episode


kelvinhours_PPO_continuous_no_price_forecast = calculate_kelvinhours(
    intermediate_model_path_PPO_no_price_forecast,
    continuous=True,
    price_forecast=False,
)
kelvinhours_DQN_linear = calculate_kelvinhours(
    intermediate_model_path_DQN_piecewise_linear
)
kelvinhours_DQN_quadratic = calculate_kelvinhours(
    intermediate_model_path_DQN_piecewise_quadratic
)
kelvinhours_PPO_linear = calculate_kelvinhours(
    intermediate_model_path_PPO_piecewise_linear,
    continuous=True,
)
kelvinhours_PPO_quadratic = calculate_kelvinhours(
    intermediate_model_path_PPO_piecewise_quadratic, continuous=True
)
kelvinhours_mpc = np.mean(
    np.abs(np.load("../mpc/mpc_temp_full_data_18.npy") - 21.0) * 0.25
)  # 0.17593532855323116

width = 432.48195
fig, ax = plt.subplots(1, 1, figsize=set_size(width))  # , sharex=True)
ax.plot(
    [i for (i, x) in kelvinhours_PPO_continuous_no_price_forecast[0]],
    [x for (i, x) in kelvinhours_PPO_continuous_no_price_forecast[0]],
    label="PPO without price forecast",
)
ax.axvline(
    kelvinhours_PPO_continuous_no_price_forecast[1], color=color_palette[0], ls=":"
)
ax.plot(
    [i for (i, x) in kelvinhours_DQN_linear[0]],
    [x for (i, x) in kelvinhours_DQN_linear[0]],
    label="DQN piecewise linear",
)
ax.axvline(kelvinhours_DQN_linear[1], color=color_palette[1], ls=":")
ax.plot(
    [i for (i, x) in kelvinhours_DQN_quadratic[0]],
    [x for (i, x) in kelvinhours_DQN_quadratic[0]],
    label="DQN piecewise quadratic",
)
ax.axvline(kelvinhours_DQN_quadratic[1], color=color_palette[2], ls=":")
ax.plot(
    [i for (i, x) in kelvinhours_PPO_linear[0]],
    [x for (i, x) in kelvinhours_PPO_linear[0]],
    label="PPO piecewise linear",
)
ax.axvline(kelvinhours_PPO_linear[1], color=color_palette[3], ls=":")
ax.plot(
    [i for (i, x) in kelvinhours_PPO_quadratic[0]],
    [x for (i, x) in kelvinhours_PPO_quadratic[0]],
    label="PPO piecewise quadratic",
)
ax.axvline(kelvinhours_PPO_quadratic[1], color=color_palette[4], ls=":")
ax.plot(3000 * [0.008671862393915835], label="P-controller")
ax.plot(3000 * [kelvinhours_mpc], color="black", label="MPC")
ax.set_ylabel("mean kelvinhours per episode")
ax.set_yscale("log")
ax.legend(loc="lower right")
ax.set_xlabel("episode")
fig.savefig("plotting/plots/kelvinhours_price-forecast.pdf", format="pdf")
