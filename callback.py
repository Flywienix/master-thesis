import os
import time
import shutil
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import ts2xy, load_results


# callback for saving best model so far (from: https://stable-baselines.readthedocs.io/en/master/guide/examples.html)
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(
        self,
        log_dir: str,
        specific_experiment_name: str,
        check_freq: int = 16_166,
        intermediate_save: int = 100,
        verbose=1,
    ):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.specific_experiment_name = specific_experiment_name
        self.best_mean_reward = -np.inf
        # self.reward_over_time = list()
        self.last_time = time.time()
        self.episode_execution_times = list()
        self.intermediate_save = intermediate_save

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(
                load_results(self.log_dir),
                "timesteps",
            )
            if len(x) > 0:
                mean_reward = np.mean(
                    y[
                        -1:  # not mean anymore
                    ]  # before: y[-self.check_freq :] MISCONSEPTION!!! reruns necessary
                )  # mean reward of timesteps of past episode
                # self.reward_over_time += [mean_reward]
                # np.save(
                #     os.path.join(
                #         self.log_dir, self.specific_experiment_name + "_reward"
                #     ),
                #     np.array(self.reward_over_time),
                # )
                self.episode_execution_times += [time.time() - self.last_time]
                self.last_time = time.time()
                np.save(
                    os.path.join(
                        self.log_dir, self.specific_experiment_name + "_execution_times"
                    ),
                    np.array(self.episode_execution_times),
                )
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best reward: {:.2f} - Last reward per episode {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.log_dir))
                    self.model.save(
                        os.path.join(
                            self.log_dir, self.specific_experiment_name + "_model"
                        )
                    )
                    # save episode that resulted in best model in info.txt
                    with open(os.path.join(self.log_dir, "info.txt"), "r") as info_file:
                        lines = info_file.readlines()
                    lines[2] = "best episode: " + str(
                        int(self.n_calls / self.check_freq)
                    )
                    with open(os.path.join(self.log_dir, "info.txt"), "w") as info_file:
                        info_file.writelines(lines)

                # save intermediate model every self.intermdieate_save episodes
                if (self.n_calls / self.check_freq) % self.intermediate_save == 0:
                    self.model.save(
                        os.path.join(
                            self.log_dir,
                            self.specific_experiment_name
                            + "_model_"
                            + str(int(self.n_calls / self.check_freq)),
                        )
                    )
        return True
