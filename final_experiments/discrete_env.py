import gym
import numpy as np
import matplotlib.pyplot as plt


class DiscreteEURUSDTradingEnv(gym.Env):
    def __init__(
        self,
        eurusd_df_observations,
        eurusd_df_real_prices,
        window_len=10,
        leverage=1,
        positive_multiplier=100,
        negative_multiplier=100,
        discrete_actions=2,
    ):
        self.eurusd_real_prices = eurusd_df_real_prices
        self.eurusd_df_features = eurusd_df_observations
        self.features = eurusd_df_observations.shape[1]
        self.window_len = window_len
        self.action_space = gym.spaces.Discrete(discrete_actions)
        low = eurusd_df_observations.min(axis=0)
        low = np.tile(low, (window_len, 1))
        high = eurusd_df_observations.max(axis=0)
        high = np.tile(high, (window_len, 1))
        self.observation_space = gym.spaces.Box(low=low, high=high)
        self.leverage = leverage
        self.positive_multiplier = positive_multiplier
        self.negative_multiplier = negative_multiplier
        self.possible_actions = np.linspace(0, 1, discrete_actions)

        # episode variables
        self.episode_tick = 0
        self.cash = 1.0
        self.eurusd = 0.0
        self.borrowed = 0.0
        self.portfolio_value = 1.0
        self.end_tick = eurusd_df_observations.shape[0] - 1 - window_len
        self.done = False
        self.previous_portfolio_value = 1.0
        self.previous_action = None

        self.in_cash = []
        self.in_eurusd = []
        self.portfolio_values = []

    def reset(self):
        self.episode_tick = 0
        self.cash = 1.0
        self.eurusd = 0.0
        self.portfolio_value = 1.0
        self.done = False
        self.previous_portfolio_value = 1.0
        self.previous_action = None
        self.borrowed = 0.0

        self.in_cash = []
        self.in_eurusd = []
        self.portfolio_values = []

        return self.get_observation()

    def step(self, action):
        self.episode_tick += 1
        # perform action
        self.perform_action(action)

        # calculate the portfolio after whole day of trading so adjust the portfolio value by close price
        self.calculate_portfolio_value_after_close()

        # calculate reward
        reward = self.calculate_reward()

        observation = self.get_observation()

        done = self.episode_tick == self.end_tick

        info = {"episode_reward": reward}

        return observation, reward, done, info

    def get_observation(self):
        return self.eurusd_df_features.iloc[
            self.episode_tick : self.episode_tick + self.window_len
        ].to_numpy()

    def perform_action(self, action):
        # calculate portfolio value
        self.portfolio_value = self.cash + self.eurusd - self.borrowed
        # self.eurusd = self.portfolio_value * action
        # money to put in from us
        if action != self.previous_action:
            our_money = self.portfolio_value * self.possible_actions[action]
            # money to put in from broker
            broker_money = our_money * (self.leverage - 1)
            # money to borrow from broker
            self.borrowed = broker_money
            self.eurusd = our_money + broker_money
            self.cash = self.portfolio_value - our_money
            self.in_cash.append(self.cash)
            self.in_eurusd.append(our_money)
        else:
            self.in_cash.append(self.cash)
            self.in_eurusd.append(self.eurusd - self.borrowed)

    def calculate_portfolio_value_after_close(self):
        self.previous_portfolio_value = self.portfolio_value
        self.eurusd *= (
            self.eurusd_real_prices.iloc[self.episode_tick + self.window_len]["close"]
            / self.eurusd_real_prices.iloc[self.episode_tick + self.window_len]["open"]
        )
        # if eur usd is euqal to broker money then we have to pay back the broker and lose our money
        if self.eurusd <= self.borrowed:
            self.eurusd = 0.0
            self.borrowed = 0.0
        self.portfolio_value = self.cash + self.eurusd - self.borrowed
        self.portfolio_values.append(self.portfolio_value)

    def calculate_reward(self):
        # calculate reward as difference between portfolio value after close and portfolio value before open
        percentage_change = float(
            (self.portfolio_value - self.previous_portfolio_value)
            / self.previous_portfolio_value
        )

        if percentage_change > 0:
            return percentage_change * self.positive_multiplier
        else:
            return percentage_change * self.negative_multiplier

    def render(self, mode="human"):
        print(f"Episode tick: {self.episode_tick}")
        print(f"Portfolio value: {self.portfolio_value}")
        print(f"EURUSD: {self.eurusd}")
        print(f"Cash: {self.cash}")
        print(f"Done: {self.done}")

    def plot_portfolio_and(self):
        # plot close prices
        plt.plot(self.eurusd_real_prices["close"], label="EURUSD real close")
        # plot portfolio values, SP500 and cash
        plt.plot(self.portfolio_values, label="Portfolio")
        plt.plot(self.in_eurusd, label="EURUSD in portfolio")
        plt.plot(self.in_cash, label="Cash in portfolio")
        plt.legend()
        plt.show()
