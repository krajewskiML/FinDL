import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
from typing import List


class ExpandCNN(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        input_channels: int,
        channel_multipliers: List[int],
        initial_window_length: int,
    ):
        super().__init__()
        current_channels = input_channels
        self.input_channels = input_channels
        self.list = nn.Sequential()
        self.final_window_length = initial_window_length
        for idx, mult in enumerate(channel_multipliers):
            self.list.add_module(
                f"conv_{idx}_{mult}",
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=current_channels * mult,
                    kernel_size=kernel_size,
                    groups=current_channels,
                ),
            )
            self.list.add_module(f"leaku_relu_{idx}", nn.LeakyReLU())
            current_channels *= mult
            self.final_window_length -= kernel_size - 1

        self.final_channels = current_channels

    def forward(self, x):
        return self.list(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ShrinkCNN(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        input_channels: int,
        channel_divisors: List[int],
        initial_window_length: int,
    ):
        super().__init__()
        current_channels = input_channels
        self.input_channels = input_channels
        self.final_window_length = initial_window_length
        self.shrinks = nn.Sequential()
        for idx, divisor in enumerate(channel_divisors):
            self.shrinks.add_module(
                f"conv_{idx}_{divisor}",
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=current_channels // divisor,
                    kernel_size=kernel_size,
                    groups=current_channels // divisor,
                ),
            )
            self.shrinks.add_module(f"leaku_relu_{idx}", nn.LeakyReLU())
            current_channels //= divisor
            self.final_window_length -= kernel_size - 1
        self.final_channels = current_channels

    def forward(self, x):
        return self.shrinks(x)


class EpisodeSummarizer(nn.Module):
    """This module is responsible for extracting information from every single channel of the input tensor.
    During initialization it is given desired amount of episodes to extract from the input tensor.
    """

    def __init__(self, episodes: int, input_channels: int, window_length: int):
        super().__init__()
        self.episodes = episodes
        self.input_channels = input_channels

        # calculating padding for the left side as we need to have length divisible by episodes
        left_padding = (
            0 if window_length % episodes == 0 else episodes - window_length % episodes
        )
        self.padding = nn.ReplicationPad1d((left_padding, 0))
        self.final_length = window_length + left_padding
        # we initialize the weights for the pairwise multiplication with values of normal
        self.block_multiplier = nn.parameter.Parameter(
            torch.Tensor(input_channels, self.final_length)
        )
        # initialize the weights for the pairwise multiplication with values of normal distribution
        nn.init.normal_(self.block_multiplier, mean=0, std=0.15)

    def forward(self, x):
        x = self.padding(x)
        x = x * self.block_multiplier
        # we reshape the data such that we can extract the episodes using mean
        x = x.reshape(x.size(0), self.input_channels, self.episodes, -1)
        x = x.mean(dim=3)
        return x


class KernelBranch(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        input_channels: int,
        channel_multipliers: List[int],
        channel_divisors: List[int],
        episodes: int,
        window_length: int,
    ):
        super().__init__()
        expand_module = ExpandCNN(
            kernel_size, input_channels, channel_multipliers, window_length
        )
        shrink_module = ShrinkCNN(
            kernel_size,
            expand_module.final_channels,
            channel_divisors,
            expand_module.final_window_length,
        )
        summarizer_module = EpisodeSummarizer(
            episodes, shrink_module.final_channels, shrink_module.final_window_length
        )
        self.final_channels = shrink_module.final_channels
        self.list = nn.Sequential(expand_module, shrink_module, summarizer_module)

    def forward(self, x):
        return self.list(x)


class BlockCombiner(nn.Module):
    def __init__(
        self,
        kernel_sizes: List[int],
        input_channels: int,
        expandors: List[int],
        shrinkers: List[int],
        desired_summary: int,
        window_length: int,
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.branches.append(
                KernelBranch(
                    kernel_size,
                    input_channels,
                    expandors,
                    shrinkers,
                    desired_summary,
                    window_length,
                )
            )
        self.final_channels = self.branches[0].final_channels

    def forward(self, x):
        # we stack the results of the branches and take the mean
        return torch.stack([branch(x) for branch in self.branches]).mean(dim=0)


class TimeSeriesFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        kernel_sizes: List[int] = [3, 5, 7],
        expandors: List[int] = [2, 2, 2],
        shrinkers: List[int] = [2, 2, 2],
        desired_summary: int = 4,
    ):
        # setting features dim to 1 as we will flatten the output of the feature extractor as
        # we can not initialize any layers before the super().__init__ call
        super(TimeSeriesFeatureExtractor, self).__init__(
            observation_space, features_dim=1
        )
        self.input_channels = observation_space.shape[1]
        self.window_length = observation_space.shape[0]

        self.feature_extractor = BlockCombiner(
            kernel_sizes,
            self.input_channels,
            expandors,
            shrinkers,
            desired_summary,
            self.window_length,
        )
        final_channels = self.feature_extractor.final_channels
        features_dim = final_channels * desired_summary
        # we set the features dim to the correct value
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(
            observations.reshape(
                (observations.shape[0], self.input_channels, self.window_length)
            )
        ).flatten(1)
