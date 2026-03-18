import os
import math
import gymnasium as gym
import numpy as np
import scipy
from reward import *
from stable_baselines3 import DQN, PPO


class BuildingEnv(gym.Env):
    def __init__(
        self,
        discrete_action_space: bool,
        building_model_index: int,
        reward_temp_term_function: callable,
        num_temperature_observations: int = 5,
        num_actions: int = 5,
        price_forecast: bool = False,
        climate_zone: str = "Csb",
        temperature_weight: float = 1_000,
        pricing_weight: float = 1,
        target_tempearture: float = 21.0,
        max_timesteps: int = 16_166,
        data_location: str = "data/",
        sin_price_function: bool = False,
        randomly_initialized_temperatures: bool = True,
        base_agent_path: (
            str | None
        ) = None,  # None means no independent learners transfer learning
        training: bool = False,
        validation: bool = False,
        result_path: str = "",
    ):
        super(BuildingEnv, self).__init__()
        self.price_forecast = price_forecast
        self.independent_learners = base_agent_path is not None
        num_price_forecast = 5 if self.price_forecast else 0
        num_independent_learners_suggestion = 1 if self.independent_learners else 0
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                num_temperature_observations
                + num_price_forecast
                + num_independent_learners_suggestion,
            ),
            dtype=np.float64,
        )
        self.action_space = (
            gym.spaces.Discrete(num_actions)
            if discrete_action_space
            else gym.spaces.Box(low=0.0, high=1.0, dtype=np.float64)
        )
        self.discrete_action_space = discrete_action_space
        self.temperature_weight = temperature_weight
        self.pricing_weight = pricing_weight
        self.target_temperature = target_tempearture
        self.max_timesteps = max_timesteps if not validation else 8760
        self.building_model = self._load_building_model(
            data_location + "models_multi_matrices.mat", building_model_index
        )
        self.price = (
            np.load(data_location + "price_data.npy")
            if not sin_price_function
            else (
                np.sin(
                    np.linspace(0, 16_263, 16_263, endpoint=False)
                    / (96 / (2 * math.pi))
                )
                * 100
                + 100
            )
        )
        if validation:
            self.price = np.load(data_location + "price_val.npy")
        self.sol = np.load(data_location + "sol_" + climate_zone + ".npy")
        if validation:
            self.sol = np.load(data_location + "sol_val.npy")
        self.Tamb = (
            np.load(data_location + "Tamb_" + climate_zone + ".npy")
            if not sin_price_function
            else np.full(np.load("data/Tamb_" + climate_zone + ".npy").shape, 5.0)
        )
        if validation:
            self.Tamb = np.load(data_location + "Tamb_val.npy")
        self.randomly_initialized_temperatures = randomly_initialized_temperatures
        self.reward_function = heatpump_reward_function(
            temperature_weight,
            pricing_weight if self.price_forecast else 0,
            target_tempearture,
            reward_temp_term_function,
        )
        if self.independent_learners:
            if "DQN" in base_agent_path:
                self.base_agent = DQN.load(
                    base_agent_path, env=None, verbose=1, device="cuda"
                )
            elif "PPO" in base_agent_path:
                self.base_agent = PPO.load(
                    base_agent_path, env=None, verbose=1, device="cpu"
                )
            else:
                raise Exception(
                    "Cannot infer method for independent learners transfer learning suggestion"
                )
        self.reward_over_time = list()
        self.temp_reward_over_time = list()
        self.price_reward_over_time = list()
        self.training = training
        self.result_path = result_path

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.u = np.zeros([3, self.max_timesteps])
        self.x = np.ones([self.observation_space.shape[0], self.max_timesteps + 1])
        initial_indoor_temperatures = (
            np.random.uniform(
                low=self.target_temperature - 1.0,
                high=self.target_temperature + 1,
                size=5,
            )
            if self.randomly_initialized_temperatures
            else [21, 21, 21, 21, 21]
        )
        self.x[: 10 if self.price_forecast else 5, 0] = (
            np.append(initial_indoor_temperatures, self.price[: 96 + 1 : 4 * 6])
            if self.price_forecast
            else initial_indoor_temperatures
        )
        self.y = np.append(
            self.building_model["C"] @ self.x[:5, self.timestep],
            np.zeros([self.max_timesteps]),
        )
        self.actions_over_time = np.zeros([self.max_timesteps])
        self.reward_per_episode = np.zeros([self.max_timesteps])
        self.temp_reward_per_episode = np.zeros([self.max_timesteps])
        self.price_reward_per_episode = np.zeros([self.max_timesteps])
        return self.x[:, self.timestep], dict()

    def step(self, action: int | float):
        self.u[:, self.timestep] = [
            self._action_to_heating_level(action),
            self.sol[self.timestep],
            self.Tamb[self.timestep],
        ]
        self.x[:5, self.timestep + 1] = (
            self.building_model["A"] @ self.x[:5, self.timestep]
            + self.building_model["B"] @ self.u[:, self.timestep]
        )
        if self.price_forecast:
            self.x[5:10, self.timestep + 1] = self.price[
                self.timestep + 1 : 96 + self.timestep + 1 + 1 : 4 * 6
            ]
        self.y[self.timestep + 1] = (
            self.building_model["C"] @ self.x[:5, self.timestep + 1]
        )[0]
        self.actions_over_time[self.timestep] = self._action_to_heating_level(action)
        reward_tuple = self.reward_function(
            self.y[self.timestep + 1],
            self.price[self.timestep],
            self.u[0, self.timestep],
        )
        reward_temp, reward_price = reward_tuple
        reward = reward_temp + reward_price
        if self.training:
            self.reward_per_episode[self.timestep] = reward
            self.temp_reward_per_episode[self.timestep] = reward_temp
            self.price_reward_per_episode[self.timestep] = reward_price
            if self.timestep == self.max_timesteps - 1:
                self.reward_over_time += [np.mean(self.reward_per_episode)]
                self.temp_reward_over_time += [np.mean(self.temp_reward_per_episode)]
                self.price_reward_over_time += [np.mean(self.price_reward_per_episode)]
                np.save(
                    os.path.join(
                        self.result_path, self.result_path.split("/")[-1] + "_reward"
                    ),
                    np.array(self.reward_over_time),
                )
                np.save(
                    os.path.join(
                        self.result_path,
                        self.result_path.split("/")[-1] + "_temp_reward",
                    ),
                    np.array(self.temp_reward_over_time),
                )
                np.save(
                    os.path.join(
                        self.result_path,
                        self.result_path.split("/")[-1] + "_price_reward",
                    ),
                    np.array(self.price_reward_over_time),
                )
        self.timestep += 1
        if self.independent_learners:
            heating_suggestion, _state = self.base_agent.predict(
                self.x[: 10 if self.price_forecast else 5, self.timestep]
            )
            heating_suggestion = self._action_to_heating_level(heating_suggestion)
            self.x[10 if self.price_forecast else 5, self.timestep] = heating_suggestion
        return (
            self.x[:, self.timestep],
            reward,
            self.timestep == self.max_timesteps,
            self.timestep == self.max_timesteps,
            {
                "state_over_time": self.x,
                "actions_over_time": self.actions_over_time,
                "price_over_time": self.price,
            },
        )

    def _action_to_heating_level(self, action: int) -> float:
        if not self.discrete_action_space:
            return action[0]
        num_actions = self.action_space.n
        return action / (num_actions - 1)

    @staticmethod
    def _load_building_model(path: str, building_model_index: int) -> dict:
        mat = scipy.io.loadmat(path)
        models = mat["models_multi_matrices"]["models"][0, 0]
        model = models[0][building_model_index][0][0]
        return dict(A=model[0], B=model[1], C=model[2])
