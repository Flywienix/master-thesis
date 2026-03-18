import os

import numpy as np
import matplotlib.pyplot as plt

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")

intermediate_model_path_DQN = (
    "runs/method-comparison/DQN_quadratic_18_w1d5_3/DQN_quadratic_18_w1d5_3_model_"
)
intermediate_model_path_PPO_discrete = (
    "runs/method-comparison/PPO_quadratic_18_w1d5_1/PPO_quadratic_18_w1d5_1_model_"
)
intermediate_model_path_PPO_continuous = (
    "runs/method-comparison/PPO_quadratic_18_w1c_4/PPO_quadratic_18_w1c_4_model_"
)


def calculate_total_cost(
    path: str, continuous: bool = False, calculate_intermediate: bool = False
) -> list:
    path_parts = path.split("/")
    model_folder = "/".join(path_parts[:-1])
    intermediate_evaluation_folder = model_folder + "/intermediate_evaluation"
    intermediate_price_action = list()
    for i in range(
        1, 30 + 1
    ):  # would need to change for training duration other than 50_000_000 timesteps
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
            )
            os.system(call)
        actions = np.load(
            intermediate_evaluation_folder
            + "/"
            + str(i * 100)
            + "/"
            + path_parts[-2]
            + "_actions.npy"
        )
        price = np.load(
            intermediate_evaluation_folder
            + "/"
            + str(i * 100)
            + "/"
            + path_parts[-2]
            + "_price.npy"
        )[: actions.shape[0]]

        intermediate_price_action.append((i * 100, price, actions))
    with open(model_folder + "/info.txt", "r") as info_file:
        best_episode = int(info_file.readlines()[-1].split(" ")[-1])
    intermediate_price_action.insert(
        int(best_episode / 100),
        (
            best_episode,
            np.load(path[:-6] + "price.npy")[: actions.shape[0]],
            np.load(path[:-6] + "actions.npy"),
        ),
    )
    total_cost = [
        (i, np.sum(price * action)) for (i, price, action) in intermediate_price_action
    ]
    return total_cost


total_cost_DQN = calculate_total_cost(intermediate_model_path_DQN)
total_cost_PPO_discrete = calculate_total_cost(intermediate_model_path_PPO_discrete)
total_cost_PPO_continuous = calculate_total_cost(
    intermediate_model_path_PPO_continuous, True
)

width = 432.48195
fig, ax = plt.subplots(1, 1, figsize=set_size(width))  # , sharex=True)
ax.plot(
    [i for (i, x) in total_cost_DQN],
    [x for (i, x) in total_cost_DQN],
    label="DQN",
)
ax.plot(
    [i for (i, x) in total_cost_PPO_discrete],
    [x for (i, x) in total_cost_PPO_discrete],
    label="PPO discrete",
)
ax.plot(
    [i for (i, x) in total_cost_PPO_continuous],
    [x for (i, x) in total_cost_PPO_continuous],
    label="PPO continuous",
)
ax.set_ylabel("heating cost per episode")
ax.legend()
ax.set_xlabel("episode")
fig.savefig("plotting/plots/total_cost_method-comparison.pdf", format="pdf")
