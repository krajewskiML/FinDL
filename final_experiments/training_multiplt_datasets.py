from training_multiple_configs import load_data, SaveOnMostProfitCallback, create_agent
from environments.utils import get_feature_set
from environments.environment import EURUSDTradingEnv
import pandas as pd

BEST_CURRENT_CONFIG = {
    "gamma": 0.99,
    "kernel_sizes": [3, 5, 7],
    "expandors": [2, 2],
    "shrinkers": [2, 2],
    "desired_summary": 8,
    "leverage": 10,
    "window_len": 120,
}

def run_specific_config(config: int, data_obs: pd.DataFrame, data_real: pd.DataFrame):
    # get appropriate data
    proper_data_obs = get_feature_set(data_obs, config)
    proper_data_real = get_feature_set(data_real, config)

    # create environments
    train_env = EURUSDTradingEnv(
        proper_data_obs.iloc[:int(len(proper_data_obs) * 0.8)],
        proper_data_real.iloc[:int(len(proper_data_real) * 0.8)],
        BEST_CURRENT_CONFIG["window_len"],
        BEST_CURRENT_CONFIG["leverage"],
        100,
        100,
    )
    test_env = EURUSDTradingEnv(
        proper_data_obs.iloc[int(len(proper_data_obs) * 0.8):],
        proper_data_real.iloc[int(len(proper_data_real) * 0.8):],
        BEST_CURRENT_CONFIG["window_len"],
        BEST_CURRENT_CONFIG["leverage"],
        100,
        100,
    )

    tensorboard_log = f"different_datasets/config_{config}"
    model_log = f"different_datasets/config_{config}/model"
    agent = create_agent(
        gamma=BEST_CURRENT_CONFIG["gamma"],
        kernel_sizes=BEST_CURRENT_CONFIG["kernel_sizes"],
        expandors=BEST_CURRENT_CONFIG["expandors"],
        shrinkers=BEST_CURRENT_CONFIG["shrinkers"],
        desired_summary=BEST_CURRENT_CONFIG["desired_summary"],
        env=train_env,
        path_to_tensorboard_logs=tensorboard_log,
    )

    save_on_most_profit_callback = SaveOnMostProfitCallback(
        eval_env=test_env,
        best_model_save_path=model_log,
        eval_freq=10_000,
    )

    agent.learn(
        total_timesteps=650_000,
        callback=save_on_most_profit_callback
    )

def run_data_configs():
    scaled, non_scaled = load_data()

    for i in range(1, 6):
        run_specific_config(i, scaled, non_scaled)


if __name__ == "__main__":
    run_data_configs()


