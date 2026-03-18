import os
import argparse

import gymnasium as gym
from env import BuildingEnv
from stable_baselines3 import DQN, PPO
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "model", help="path to model that will be evaluated", metavar="PATH"
)
parser.add_argument("building_model", help="index of building model to train on")
parser.add_argument(
    "-o",
    "--output_folder",
    help="the folder where the scripts output is saved (model folder if not set)",
)
parser.add_argument(
    "-c",
    "--continuous_action_space",
    action="store_true",
    help="set continuous action space (does not work with every method)",
)
parser.add_argument(
    "-p",
    "--price_forecast",
    action="store_true",
    help="give price forecast to agent",
)
parser.add_argument(
    "-v", "--validation", action="store_true", help="use validation data for evaluation"
)
args = parser.parse_args()

gym.register(id="BuildingEnv", entry_point=BuildingEnv)
env = gym.make(
    "BuildingEnv",
    discrete_action_space=not args.continuous_action_space,
    building_model_index=int(args.building_model),
    reward_temp_term_function=lambda x: x,  # irrelevant for evaluation
    price_forecast=args.price_forecast,
    randomly_initialized_temperatures=False,
    validation=args.validation,
)
if "DQN" in args.model:
    model = DQN.load(args.model, env, verbose=1, device="cuda")
elif "PPO" in args.model:
    model = PPO.load(args.model, env, verbose=1, device="cpu")
else:
    raise Exception("Cannot infer method")

vec_env = model.get_env()
obs = vec_env.reset()
done = False
info = dict()
while not done:
    action, _state = model.predict(obs)
    obs, _reward, done, info = vec_env.step(action)

output_folder = (
    args.output_folder if args.output_folder else "/".join(args.model.split("/")[0:-1])
)
experiment_name = args.model.split("/")[-2]

temperature_over_time = info[0]["state_over_time"][0]
actions_over_time = info[0]["actions_over_time"]
price_over_time = info[0]["price_over_time"]

if args.validation:
    np.save(
        os.path.join(output_folder, experiment_name + "_temp_val"),
        temperature_over_time,
    )
    np.save(
        os.path.join(output_folder, experiment_name + "_actions_val"), actions_over_time
    )
    np.save(
        os.path.join(output_folder, experiment_name + "_price_val"), price_over_time
    )

else:
    np.save(
        os.path.join(output_folder, experiment_name + "_temp"), temperature_over_time
    )
    np.save(
        os.path.join(output_folder, experiment_name + "_actions"), actions_over_time
    )
    np.save(os.path.join(output_folder, experiment_name + "_price"), price_over_time)
