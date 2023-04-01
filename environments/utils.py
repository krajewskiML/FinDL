import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sequitur.models import LINEAR_AE, LSTM_AE
from sequitur import quick_train
from sequitur.quick_train import train_model
import yfinance as yf
import pandas as pd
from finta import TA
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import gc

def create_ta_features(ticker='^GSPC', start_='2010-01-01', end_='2022-12-31', interval_='1d', fillna=True, scale_to_std=True, fill_weekends=True):
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
    df.rename(columns={"Open": "open", "Adj Close": "close", "High": "high", "Low": "low", "Volume": "volume"}, inplace=True)
    # drop close column
    df.drop("Close", inplace=True, axis=1)
    # fill weekends
    if fill_weekends:
        df = df.resample('D').ffill()
    # get all functions in finta
    finta_functions = [func for func in dir(TA) if callable(getattr(TA, func)) and not func.startswith("__")]
    # loop through all functions in finta and append the results to the dataframe
    # skip functions that throw errors
    for func in finta_functions:
        try:
            df[func] = getattr(TA, func)(df)
        except:
            pass
    # fill in missing values
    if fillna:
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
    # scale to standard deviation, by column
    if scale_to_std:
        df = (df - df.mean()) / df.std()
    return df

def add_time_features(df):
    """
    Adds sine and cosine of weekday, monthday, yearday to dataframe
    :param df: dataframe to add time features to
    :return: dataframe with time features
    """
    # get frequency of data
    freq = pd.infer_freq(df.index)
    # if frequency is daily, assume data includes weekends
    if freq == 'D':
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
    df['weekday'] = df.index.dayofweek
    df['monthday'] = df.index.day
    df['yearday'] = df.index.dayofyear
    # add sine and cosine of weekday, monthday, yearday features
    df['sin_weekday'] = np.sin(2 * np.pi * df['weekday'] / days_in_week)
    df['cos_weekday'] = np.cos(2 * np.pi * df['weekday'] / days_in_week)
    df['sin_monthday'] = np.sin(2 * np.pi * df['monthday'] / days_in_month)
    df['cos_monthday'] = np.cos(2 * np.pi * df['monthday'] / days_in_month)
    df['sin_yearday'] = np.sin(2 * np.pi * df['yearday'] / days_in_year)
    df['cos_yearday'] = np.cos(2 * np.pi * df['yearday'] / days_in_year)
    # drop weekday, monthday, yearday features
    df.drop(['weekday', 'monthday', 'yearday'], inplace=True, axis=1)
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
        windows.append(df.iloc[i:i + window_size])
    return windows