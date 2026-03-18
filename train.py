import os
import random
import argparse

import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch

from env import BuildingEnv
from reward import piecewise_linear, piecewise_quadratic
from callback import SaveOnBestTrainingRewardCallback

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("method", help="method for training (DQN, PPO)")
parser.add_argument(
    "reward_temperature_term",
    help="how to calculate temperature term of the reward function (quadratic, quartic, piecewise_linear, piecewise_quadratic)",
)
parser.add_argument("building_model", help="index of building model to train on")
parser.add_argument(
    "-w",
    "--temperature_weight",
    help="set weight for the temperature term in the reward function (default value is 1000)",
)
parser.add_argument(
    "-p",
    "--price_forecast",
    action="store_true",
    help="give price forecast to agent for training",
)
parser.add_argument(
    "-c",
    "--continuous_action_space",
    action="store_true",
    help="set continuous action space (does not work with every method)",
)
parser.add_argument(
    "-t",
    "--transfer",
    help="load a model from a path for transfer learning",
    metavar="PATH",
)
parser.add_argument(
    "-i",
    "--independent_learners",
    help="load a model from a path for independent learners transfer learning",
    metavar="PATH",
)  # TODO make necessary changes to environment
parser.add_argument("-f", "--sub_folder", help="sub-folder of where to put results")
parser.add_argument("-s", "--seed", help="set a seed for reproducibility")
parser.add_argument(
    "-m",
    "--machine",
    help="optional info of what machine this run was executed on (home, laptop, hpc) for info.txt",
)
args = parser.parse_args()
if args.method not in ["DQN", "PPO"]:
    raise Exception("unknown method provided in arguments")
if args.reward_temperature_term not in [
    "quadratic",
    "quartic",
    "piecewise_linear",
    "piecewise_quadratic",
]:
    raise Exception("unknown temperature term function for reward function")
if args.continuous_action_space and args.method in [
    "DQN"
]:  # methods that do not support continuous action space
    raise Exception("method does not support continuous action space")

# setup result-directory
log_dir = os.path.join("runs/", args.sub_folder if args.sub_folder else "")
os.makedirs(log_dir, exist_ok=True)
num_folders = len(
    os.listdir(os.path.join("runs/", args.sub_folder if args.sub_folder else ""))
)
specific_experiment_name = (
    args.method
    + "_"
    + args.reward_temperature_term
    + "_"
    + args.building_model
    + "_"
    + "w"
    + (args.temperature_weight if args.temperature_weight else "1000")
    + ("p" if args.price_forecast else "")
    + ("c" if args.continuous_action_space else "d5")  # d5 for discrete with 5 actions
    + ("t" if args.transfer else "")
    + ("i" if args.independent_learners else "")
    + "_"
    + str(num_folders + 1)
)
log_dir = os.path.join(log_dir, specific_experiment_name)
os.makedirs(log_dir, exist_ok=True)

# set seeds for reproducibilty
random_seed = random.randint(0, 2**32 - 1)
if args.seed:
    random_seed = int(args.seed)
print("Seed:", random_seed)
# https://stackoverflow.com/questions/79001080/stable-baselines-3-the-reason-of-fixing-seed-when-load-the-model
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# create info file that stores  this run.
file = open(os.path.join(log_dir, "info.txt"), "w")
file.write(
    "python train.py "
    + args.method
    + " "
    + args.reward_temperature_term
    + " "
    + args.building_model
    + (
        " --temperature_weight " + args.temperature_weight
        if args.temperature_weight
        else ""
    )
    + (" --price_forecast" if args.price_forecast else "")
    + (" --continuous_action_space" if args.continuous_action_space else "")
    + (" --transfer " + args.transfer if args.transfer else "")
    + (
        " --independent_learners " + args.independent_learners
        if args.independent_learners
        else ""
    )
    + (" --sub_folder " + args.sub_folder if args.sub_folder else "")
    + " --seed "
    + str(random_seed)
    + ("\nsystem: " + args.machine if args.machine else "\n")
    + "\nbest episode: "
)
file.close()

# setup reward, environment and model
term_function_dict = {
    "quadratic": lambda x: x**2,
    "quartic": lambda x: x**4,
    "piecewise_linear": piecewise_linear,
    "piecewise_quadratic": piecewise_quadratic,
}
reward_temperature_term = term_function_dict[args.reward_temperature_term]
gym.register(id="BuildingEnv", entry_point=BuildingEnv)
env = gym.make(
    "BuildingEnv",
    price_forecast=args.price_forecast,
    discrete_action_space=not args.continuous_action_space,
    temperature_weight=(
        float(args.temperature_weight) if args.temperature_weight else 1_000
    ),
    building_model_index=int(args.building_model),
    reward_temp_term_function=reward_temperature_term,
    base_agent_path=args.independent_learners if args.independent_learners else None,
    training=True,
    result_path=log_dir,
)
env = Monitor(env, log_dir)
if args.transfer:
    if args.method == "DQN":
        model = DQN.load(args.transfer, env, verbose=1, device="cuda", seed=random_seed)
    if args.method == "PPO":
        model = PPO.load(args.transfer, env, verbose=1, device="cpu", seed=random_seed)
else:
    if args.method == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, device="cuda", seed=random_seed)
    if args.method == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, device="cpu", seed=random_seed)

# training
callback = SaveOnBestTrainingRewardCallback(log_dir, specific_experiment_name)
model.learn(total_timesteps=50_000_000, callback=callback)
