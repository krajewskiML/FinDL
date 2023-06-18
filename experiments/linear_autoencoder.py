import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import pandas as pd
from finta import TA
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import gc


def create_ta_features(
    ticker="^GSPC",
    start_="2010-01-01",
    end_="2022-12-31",
    interval_="1d",
    fillna=True,
    scale_to_std=True,
    fill_weekends=True,
):
    """
    Creates dataframe with technical analysis features
    :param ticker: ticker symbol to download data for (default is S&P 500)
    :param start_: start date
    :param end_: end date
    :param interval_: data frequency
    :param fillna: whether to fill in missing values
    :param scale_to_std: whether to scale to standard deviation
    :param fill_weekends: whether to fill in weekends
    :return: dataframe with technical analysis features
    """
    # download data
    df = yf.download(ticker, start_, end_, interval=interval_)
    # rename columns
    df.rename(
        columns={
            "Open": "open",
            "Adj Close": "close",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
        },
        inplace=True,
    )
    # drop close column
    df.drop("Close", inplace=True, axis=1)
    # fill weekends
    if fill_weekends:
        df = df.resample("D").ffill()
    # get all functions in finta
    finta_functions = [
        func
        for func in dir(TA)
        if callable(getattr(TA, func)) and not func.startswith("__")
    ]
    # loop through all functions in finta and append the results to the dataframe
    # skip functions that throw errors
    for func in finta_functions:
        try:
            df[func] = getattr(TA, func)(df)
        except:
            pass
    # fill in missing values
    if fillna:
        df.fillna(method="bfill", inplace=True)
        df.fillna(method="ffill", inplace=True)
    # scale to standard deviation, by column
    if scale_to_std:
        df = (df - df.mean()) / df.std()
    return df


# function that adds sine and cosine of weekday, monthday, yearday to dataframe
# takes into account whether the data is daily, hourly, minutely, etc.
# also takes into account whether data includes weekends or not
# if data does not include weekends, assume the week is 5 days, not 7, month is 21 days, not 31, and year is 250 days, not 365
def add_time_features(df):
    """
    Adds sine and cosine of weekday, monthday, yearday to dataframe
    :param df: dataframe to add time features to
    :return: dataframe with time features
    """
    # get frequency of data
    freq = pd.infer_freq(df.index)
    # if frequency is daily, assume data includes weekends
    if freq == "D":
        include_weekends = True
    else:
        include_weekends = False

    # get number of days in week, month, year
    if include_weekends:
        days_in_week = 7
        days_in_month = 31
        days_in_year = 365
    else:
        days_in_week = 5
        days_in_month = 21
        days_in_year = 250
    # add weekday, monthday, yearday features
    df["weekday"] = df.index.dayofweek
    df["monthday"] = df.index.day
    df["yearday"] = df.index.dayofyear
    # add sine and cosine of weekday, monthday, yearday features
    df["sin_weekday"] = np.sin(2 * np.pi * df["weekday"] / days_in_week)
    df["cos_weekday"] = np.cos(2 * np.pi * df["weekday"] / days_in_week)
    df["sin_monthday"] = np.sin(2 * np.pi * df["monthday"] / days_in_month)
    df["cos_monthday"] = np.cos(2 * np.pi * df["monthday"] / days_in_month)
    df["sin_yearday"] = np.sin(2 * np.pi * df["yearday"] / days_in_year)
    df["cos_yearday"] = np.cos(2 * np.pi * df["yearday"] / days_in_year)
    # drop weekday, monthday, yearday features
    df.drop(["weekday", "monthday", "yearday"], inplace=True, axis=1)
    return df


def sliding_window(df, window_size=10):
    """
    Creates a sliding window mechanism for a given dataframe
    :param df: dataframe
    :param window_size: window size
    :return: list of windows as dataframes
    """
    windows = []
    for i in range(len(df) - window_size + 1):
        windows.append(df.iloc[i : i + window_size])
    return windows


# define encoder class which will take in a sliding window of stock data 2d tensor and transform it into a 1d tensor
class LinearEncoder(nn.Module):
    def __init__(
        self, input_dims: Tuple[int, int], encoding_dim: int, h_dims: List[int]
    ):
        super(LinearEncoder, self).__init__()
        # define list of hidden layers
        self.h_layers = nn.ModuleList()
        # define list of hidden layer dimensions
        self.h_dims = [input_dims[0] * input_dims[1], *h_dims, encoding_dim]
        # loop through all hidden layers
        for i in range(len(self.h_dims) - 1):
            # add linear transformation to list of hidden layers
            self.h_layers.append(nn.Linear(self.h_dims[i], self.h_dims[i + 1]))
        # define output layer
        self.out_layer = nn.Linear(self.h_dims[-1], encoding_dim)

    def forward(self, x):
        # flatten input
        x = x.view(x.shape[0], -1)
        # loop through all hidden layers
        for layer in self.h_layers:
            # apply linear transformation
            x = layer(x)
            # apply activation function
            x = torch.sigmoid(x)
        # apply linear transformation to output layer
        x = self.out_layer(x)
        return x


# define decoder class which will take in a 1d tensor and transform it into a 2d tensor
class LinearDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dims: Tuple[int, int], h_dims: List[int]):
        super(LinearDecoder, self).__init__()
        # define list of hidden layers
        self.h_layers = nn.ModuleList()
        self.output_dims = output_dims
        # define list of hidden layer dimensions
        self.h_dims = [input_dim, *h_dims, output_dims[0] * output_dims[1]]
        # loop through all hidden layers
        for i in range(len(self.h_dims) - 1):
            # add linear transformation to list of hidden layers
            self.h_layers.append(nn.Linear(self.h_dims[i], self.h_dims[i + 1]))
        # define output layer
        self.out_layer = nn.Linear(self.h_dims[-1], output_dims[0] * output_dims[1])

    def forward(self, x):
        # loop through all hidden layers
        for layer in self.h_layers:
            # apply linear transformation
            x = layer(x)
            # apply activation function
            x = torch.sigmoid(x)
        # apply linear transformation to output layer
        x = self.out_layer(x)
        # reshape output
        x = x.view(x.size(0), *self.output_dims)
        return x


# define autoencoder class which will take in a sliding window of stock data 2d tensor
# and output a 2d tensor of the same size
class LinearAE(nn.Module):
    def __init__(self, input_dim, encoding_dim, h_dims):
        super(LinearAE, self).__init__()
        # define encoder
        self.encoder = LinearEncoder(input_dim, encoding_dim, h_dims)
        # define decoder
        self.decoder = LinearDecoder(encoding_dim, input_dim, h_dims)

    def forward(self, x):
        z = self.encoder(x)
        x_prime = self.decoder(z)
        return x_prime


class SlidingWindowDataset(Dataset):
    def __init__(self, df, window_size=10):
        self.df = df
        self.window_size = window_size

    def __len__(self):
        return self.df.shape[0] - self.window_size + 1

    def __getitem__(self, idx):
        return torch.tensor(
            self.df.iloc[idx : idx + self.window_size].values, dtype=torch.float32
        )


# main
if __name__ == "__main__":
    sp500_df = create_ta_features()
    sp500_df = add_time_features(sp500_df)

    all_data_dataset = SlidingWindowDataset(sp500_df, window_size=30)
    all_data_dataloader = DataLoader(all_data_dataset, batch_size=512, shuffle=False)

    linear_ae = LinearAE(
        input_dim=(30, sp500_df.shape[1]), encoding_dim=32, h_dims=[512, 128, 64]
    )
    # move model to GPU
    linear_ae = linear_ae.to("cuda")

    epochs = 200
    lr = 3e-4
    optimizer = torch.optim.Adam(linear_ae.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        for batch in all_data_dataloader:
            batch = batch.to("cuda")
            output = linear_ae(batch)
            loss = nn.functional.mse_loss(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
            torch.cuda.empty_cache()
            _ = gc.collect()

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
