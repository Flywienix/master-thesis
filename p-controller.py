import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from env import BuildingEnv

gym.register(id="BuildingEnv", entry_point=BuildingEnv)
env = gym.make(
    "BuildingEnv",
    discrete_action_space=False,
    building_model_index=18,
    reward_temp_term_function=lambda x: x,
    randomly_initialized_temperatures=False,
)
observation, info = env.reset()

kp = 6.15  # proportional gain for a simple P-controller

episode_over = False
while not episode_over:
    action = [kp * (21 - observation[0])]  # TODO: p-controller
    if action[0] >= 1:
        action = [1]
    elif action[0] <= 0:
        action = [0]
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()

temperature_over_time = info["state_over_time"][0]
actions_over_time = info["actions_over_time"]
# print(actions_over_time)
# print(temperature_over_time)
plt.plot(temperature_over_time)
plt.plot(actions_over_time + 20)
plt.legend()
# plt.show()

# calculate kelvinhours of p-controller
kelvinhours = np.mean(np.abs(temperature_over_time - 21.0) * 0.25)
print("kelvinhours:", kelvinhours)

# calculate total cost of p-controller
price_over_time = info["price_over_time"]
total_cost = np.sum(price_over_time[:16_166] * actions_over_time)
print("total cost:", total_cost)

# kelvinhours: 0.008671862393915835
# total cost: 235665.3179669784
