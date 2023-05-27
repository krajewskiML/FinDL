import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from typing import List, Tuple

from torch import nn
import torch


class Actions(Enum):
    Sell = 0
    Buy = 1


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick + 1]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError


class ForexEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, unit_side='left'):
        assert len(frame_bound) == 2
        assert unit_side.lower() in ['left', 'right']

        self.frame_bound = frame_bound
        self.unit_side = unit_side.lower()
        super().__init__(df, window_size)

        self.trade_fee = 0.0003  # unit


    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0  # pip

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Short:
                step_reward += -price_diff * 10000
            elif self._position == Positions.Long:
                step_reward += price_diff * 10000

        return step_reward


    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self.unit_side == 'left':
                if self._position == Positions.Short:
                    quantity = self._total_profit * (last_trade_price - self.trade_fee)
                    self._total_profit = quantity / current_price

            elif self.unit_side == 'right':
                if self._position == Positions.Long:
                    quantity = self._total_profit / last_trade_price
                    self._total_profit = quantity * (current_price - self.trade_fee)


    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]

            if self.unit_side == 'left':
                if position == Positions.Short:
                    quantity = profit * (last_trade_price - self.trade_fee)
                    profit = quantity / current_price

            elif self.unit_side == 'right':
                if position == Positions.Long:
                    quantity = profit / last_trade_price
                    profit = quantity * (current_price - self.trade_fee)

            last_trade_tick = current_tick - 1

        return profit


class SP500TradingEnv(gym.Env):

    def __init__(self, sp500_df_observations, sp500_df_real_prices, window_len=10):
        self.sp500_real_prices = sp500_df_real_prices
        self.sp500_df_features = sp500_df_observations
        self.features = sp500_df_observations.shape[1]
        self.window_len = window_len
        self.action_space = gym.spaces.Box(low=0, high=1, dtype=np.float32, shape=(1,))
        low = sp500_df_observations.min(axis=0)
        low = np.tile(low, (window_len, 1))
        high = sp500_df_observations.max(axis=0)
        high = np.tile(high, (window_len, 1))
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(window_len, self.features))

        # episode variables
        self.episode_tick = 0
        self.cash = 1.0
        self.sp500 = 0.0
        self.portfolio_value = 1.0
        self.end_tick = sp500_df_observations.shape[0] - window_len - 1
        self.done = False
        self.previous_portfolio_value = 1.0

        self.in_cash = []
        self.in_sp500 = []
        self.portfolio_values = []

    def reset(self):
        self.episode_tick = 0
        self.cash = 1.0
        self.sp500 = 0.0
        self.portfolio_value = 1.0
        self.done = False
        self.previous_portfolio_value = 1.0

        self.in_cash = []
        self.in_sp500 = []
        self.portfolio_values = []

        return self.get_observation()

    def step(self, action):
        # perform action
        self.perform_action(action)

        # calculate the portfolio after whole day of trading so adjust the portfolio value by close price
        self.calculate_portfolio_value_after_close()

        # calculate reward
        reward = self.calculate_reward()

        self.episode_tick += 1

        observation = self.get_observation()

        done = self.episode_tick == self.end_tick

        info = {'episode_reward': reward}

        return observation, reward, done, info


    def get_observation(self):
        return self.sp500_df_features.iloc[self.episode_tick:self.episode_tick + self.window_len].to_numpy()

    def perform_action(self, action):
        # calculate portfolio value
        self.portfolio_value = self.cash + self.sp500  # simulate cashout of SP500
        self.sp500 = self.portfolio_value * action  # simulate buying SP500
        self.cash = self.portfolio_value - self.sp500  # get cash after buying SP500
        self.in_cash.append(self.cash)
        self.in_sp500.append(self.sp500)

    def calculate_portfolio_value_after_close(self):
        self.previous_portfolio_value = self.portfolio_value
        self.sp500 *= self.sp500_real_prices.iloc[self.episode_tick + self.window_len]['close'] / self.sp500_real_prices.iloc[self.episode_tick + self.window_len]['open']
        self.portfolio_value = self.cash + self.sp500
        self.portfolio_values.append(self.portfolio_value)

    def calculate_reward(self):
        # calculate reward as difference between portfolio value after close and portfolio value before open
        return float((self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value)

    def render(self, mode="human"):
        print(f"Episode tick: {self.episode_tick}")
        print(f"Portfolio value: {self.portfolio_value}")
        print(f"SP500: {self.sp500}")
        print(f"Cash: {self.cash}")
        print(f"Done: {self.done}")

    def plot_portfolio_and(self):
        # plot close prices
        plt.plot(self.sp500_real_prices['close'], label='SP500 real close')
        # plot portfolio values, SP500 and cash
        plt.plot(self.portfolio_values, label='Portfolio')
        plt.plot(self.in_sp500, label='SP500 in portfolio')
        plt.plot(self.in_cash, label='Cash in portfolio')
        plt.legend()
        plt.show()

class SP500TradingEnvAutoencoder(SP500TradingEnv):

    def __init__(self, sp500_df_observations, sp500_df_real_prices, window_len=10, encoder: nn.Module= None, device: str = 'cuda'):
        super().__init__(sp500_df_observations, sp500_df_real_prices, window_len)
        self.encoder = encoder
        self.device = device
        self.representations = self.calculate_representations()

    def calculate_representations(self) -> List[np.ndarray]:
        values_on_gpu = torch.tensor(self.sp500_df_features.values, dtype=torch.float32).to(self.device)
        return [self.encoder(values_on_gpu[tick: tick + self.window_len].view(1, self.window_len * self.features)) for tick in range(0, self.end_tick + 1)]

    def get_observation(self):
        return self.representations[self.episode_tick]

class EURUSDTradingEnv(gym.Env):

    def __init__(self, eurusd_df_observations, eurusd_df_real_prices, window_len=10, leverage=1, positive_multiplier=100, negative_multiplier=100):
        self.eurusd_real_prices = eurusd_df_real_prices
        self.eurusd_df_features = eurusd_df_observations
        self.features = eurusd_df_observations.shape[1]
        self.window_len = window_len
        self.action_space = gym.spaces.Box(low=0, high=1, dtype=np.float32, shape=(1,))
        low = eurusd_df_observations.min(axis=0)
        low = np.tile(low, (window_len, 1))
        high = eurusd_df_observations.max(axis=0)
        high = np.tile(high, (window_len, 1))
        self.observation_space = gym.spaces.Box(low=low, high=high)
        self.leverage = leverage
        self.positive_multiplier = positive_multiplier
        self.negative_multiplier = negative_multiplier

        # episode variables
        self.episode_tick = 0
        self.cash = 1.0
        self.eurusd = 0.0
        self.borrowed = 0.0
        self.portfolio_value = 1.0
        self.end_tick = eurusd_df_observations.shape[0] - 1 - window_len
        self.done = False
        self.previous_portfolio_value = 1.0

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

        info = {'episode_reward': reward}

        return observation, reward, done, info

    def get_observation(self):
        return self.eurusd_df_features.iloc[self.episode_tick:self.episode_tick + self.window_len].to_numpy()

    def perform_action(self, action):
        # calculate portfolio value
        self.portfolio_value = self.cash + self.eurusd - self.borrowed
        # self.eurusd = self.portfolio_value * action
        # money to put in from us
        our_money = self.portfolio_value * action
        # money to put in from broker
        broker_money = our_money * (self.leverage - 1)
        # money to borrow from broker
        self.borrowed = broker_money
        self.eurusd = our_money + broker_money
        self.cash = self.portfolio_value - our_money
        self.in_cash.append(self.cash)
        self.in_eurusd.append(our_money)

    def calculate_portfolio_value_after_close(self):
        self.previous_portfolio_value = self.portfolio_value
        self.eurusd *= self.eurusd_real_prices.iloc[self.episode_tick + self.window_len]['close'] / self.eurusd_real_prices.iloc[self.episode_tick + self.window_len]['open']
        # if eur usd is euqal to broker money then we have to pay back the broker and lose our money
        if self.eurusd <= self.borrowed:
            self.eurusd = 0.0
            self.borrowed = 0.0
        self.portfolio_value = self.cash + self.eurusd - self.borrowed
        self.portfolio_values.append(self.portfolio_value)

    def calculate_reward(self):
        # calculate reward as difference between portfolio value after close and portfolio value before open
        percentage_change = float((self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value)

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
        plt.plot(self.eurusd_real_prices['close'], label='EURUSD real close')
        # plot portfolio values, SP500 and cash
        plt.plot(self.portfolio_values, label='Portfolio')
        plt.plot(self.in_eurusd, label='EURUSD in portfolio')
        plt.plot(self.in_cash, label='Cash in portfolio')
        plt.legend()
        plt.show()


