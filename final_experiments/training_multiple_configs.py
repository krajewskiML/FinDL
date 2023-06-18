from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from feature_extractor_arch import TimeSeriesFeatureExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import pandas as pd
from environments.environment import EURUSDTradingEnv
import numpy as np

from typing import Tuple
import os


class SaveOnMostProfitCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        best_model_save_path,
        verbose=0,
        eval_freq=10000,
    ):
        super(SaveOnMostProfitCallback, self).__init__(verbose)
        self.best_profit_at_the_end = 0.0
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.eval_freq = eval_freq

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            current_profit = self._get_profit()
            if current_profit > self.best_profit_at_the_end:
                self.best_profit_at_the_end = current_profit
                model_name = os.path.join(self.best_model_save_path, f"best_model_{current_profit}")
                self.model.save(model_name)
                print(f"Saved model with portfolio value: {current_profit}")
        return True

    def _get_profit(self):
        obs = self.eval_env.reset()
        done = False
        while not done:
            action, _states = self.model.predict(obs.reshape(1, self.eval_env.observation_space.shape[0], self.eval_env.observation_space.shape[1]))
            obs, rewards, done, info = self.eval_env.step(action)
        return self.eval_env.portfolio_values[-1][0][0]



def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    eur_usd_ta_scaled_tf = pd.read_csv(
        "/Users/maciejkrajewskistudia/PycharmProjects/FinDL/data/EURUSD_1m_ta_scaled_tf.csv",
        index_col=0,
        parse_dates=True,
        sep="\t",
    )

    eur_usd_ta_tf = pd.read_csv(
        "/Users/maciejkrajewskistudia/PycharmProjects/FinDL/data/EURUSD_1m_ta_tf.csv",
        index_col=0,
        parse_dates=True,
        sep="\t",
    )

    return eur_usd_ta_scaled_tf, eur_usd_ta_tf


def create_agent(
    gamma,
    kernel_sizes,
    expandors,
    shrinkers,
    desired_summary,
    env,
    path_to_tensorboard_logs
) -> PPO:
    policy_kwargs = dict(
        features_extractor_class=TimeSeriesFeatureExtractor,
        features_extractor_kwargs=dict(
            kernel_sizes=kernel_sizes,
            expandors=expandors,
            shrinkers=shrinkers,
            desired_summary=desired_summary,
        ),
    )

    return PPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        gamma=gamma,
        tensorboard_log=path_to_tensorboard_logs,
    )

def create_envs(
    window_len: int,
    leverage: int,
    positive_mult: int,
    negative_mult: int
) -> Tuple[EURUSDTradingEnv, EURUSDTradingEnv]:
    eur_usd_ta_scaled_tf, eur_usd_ta_tf = load_data()

    # split the data into train and test
    eur_usd_ta_scaled_tf_train = eur_usd_ta_scaled_tf.iloc[:int(len(eur_usd_ta_scaled_tf) * 0.8)]
    eur_usd_ta_scaled_tf_test = eur_usd_ta_scaled_tf.iloc[int(len(eur_usd_ta_scaled_tf) * 0.8):]

    eur_usd_ta_tf_train = eur_usd_ta_tf.iloc[:int(len(eur_usd_ta_tf) * 0.8)]
    eur_usd_ta_tf_test = eur_usd_ta_tf.iloc[int(len(eur_usd_ta_tf) * 0.8):]

    train_env = EURUSDTradingEnv(
        eur_usd_ta_scaled_tf_train,
        eur_usd_ta_tf_train,
        window_len,
        leverage,
        positive_mult,
        negative_mult
    )
    test_env = EURUSDTradingEnv(
        eur_usd_ta_scaled_tf_test,
        eur_usd_ta_tf_test,
        window_len,
        leverage,
        positive_mult,
        negative_mult
    )

    return train_env, test_env

def run_training():
    # define search space
    gammas = [0.7, 0.9, 0.99] # 3
    kernel_sizes = [[3, 5, 7], [3, 5]] # 2
    expandors_shrinkers = [[2, 2], [2, 2, 2]] # 2
    desired_summaries = [4, 8] # 2
    window_lens = [120, 180] # 2
    leverages = [10] # 1
    positive_mults = [100] # 1
    negative_mults = [100] # 1

    # iterate over search space
    # first we will iterate over variables that are related to the environment
    for window_len in window_lens:
        for positive in positive_mults:
            for negative in negative_mults:
                for leverage in leverages:
                    train_env, test_env = create_envs(window_len, leverage, positive, negative)
                    # then we will iterate over variables that are related to the agent
                    for gamma in gammas:
                        for kernel_size in kernel_sizes:
                            for expandors_shrinker in expandors_shrinkers:
                                for desired_summary in desired_summaries:
                                    path = os.path.join(
                                        f"window_len_{window_len}",
                                        f"positive_{positive}",
                                        f"negative_{negative}",
                                        f"leverage_{leverage}",
                                        f"gamma_{gamma}",
                                        f"kernel_size_{kernel_size}",
                                        f"expandors_shrinker_{expandors_shrinker}",
                                        f"desired_summary_{desired_summary}"
                                    )
                                    model_path = os.path.join("models", path)
                                    tensorboard_path = os.path.join("tensorboard", path)
                                    if not os.path.exists(model_path):
                                        os.makedirs(model_path)
                                    if not os.path.exists(tensorboard_path):
                                        os.makedirs(tensorboard_path)
                                    agent = create_agent(
                                        gamma,
                                        kernel_size,
                                        expandors_shrinker,
                                        expandors_shrinker,
                                        desired_summary,
                                        train_env,
                                        tensorboard_path
                                    )
                                    # create path to save the best model using variables that are related to the environment and the agent
                                    save_on_most_profit_callback = SaveOnMostProfitCallback(
                                        eval_env=test_env,
                                        best_model_save_path=model_path,
                                        eval_freq=40_000,
                                    )
                                    agent.learn(
                                        total_timesteps=241_000,
                                        callback=save_on_most_profit_callback
                                    )

if __name__ == "__main__":
    run_training()



