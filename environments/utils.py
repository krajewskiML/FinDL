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


def get_data(ticker='^GSPC', start_='2010-01-01', end_='2022-12-31', interval_='1d'):
    """
    Downloads data from Yahoo Finance and creates technical analysis features
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
    df.rename(columns={"Open": "open", "Adj Close": "close", "High": "high", "Low": "low", "Volume": "volume"},
              inplace=True)
    # drop close column
    df.drop("Close", inplace=True, axis=1)
    return df


# def create_ta_features(ticker='^GSPC', start_='2010-01-01', end_='2022-12-31', interval_='1d', fillna=True, scale_to_std=True, fill_weekends=False):
#     """
#     Creates dataframe with technical analysis features
#     :param ticker: ticker symbol to download data for (default is S&P 500)
#     :param start_: start date
#     :param end_: end date
#     :param interval_: data frequency
#     :param fillna: whether to fill in missing values
#     :param scale_to_std: whether to scale to standard deviation
#     :param fill_weekends: whether to fill in weekends
#     :return: dataframe with technical analysis features
#     """
#     # download data
#     df = get_data(ticker, start_, end_, interval_)
#     # fill weekends
#     if fill_weekends:
#         df = df.resample('D').ffill()
#     # get all functions in finta
#     finta_functions = [func for func in dir(TA) if callable(getattr(TA, func)) and not func.startswith("__")]
#     # loop through all functions in finta and append the results to the dataframe
#     # skip functions that throw errors
#     for func in finta_functions:
#         try:
#             df[func] = getattr(TA, func)(df)
#         except:
#             pass
#     # fill in missing values
#     if fillna:
#         df.fillna(method='bfill', inplace=True)
#         df.fillna(method='ffill', inplace=True)
#     # scale to standard deviation, by column
#     if scale_to_std:
#         df = (df - df.mean()) / df.std()
#     return df
#
# def add_time_features(df):
#     """
#     Adds sine and cosine of weekday, monthday, yearday to dataframe
#     :param df: dataframe to add time features to
#     :return: dataframe with time features
#     """
#     # get frequency of data
#     freq = pd.infer_freq(df.index)
#     # if frequency is daily, assume data includes weekends
#     if freq == 'D':
#         include_weekends = True
#     else:
#         include_weekends = False
#
#     # get number of days in week, month, year
#     if include_weekends:
#         days_in_week = 7
#         days_in_month = 31
#         days_in_year = 365
#     else:
#         days_in_week = 5
#         days_in_month = 21
#         days_in_year = 250
#     # add weekday, monthday, yearday features
#     df['weekday'] = df.index.dayofweek
#     df['monthday'] = df.index.day
#     df['yearday'] = df.index.dayofyear
#     # add sine and cosine of weekday, monthday, yearday features
#     df['sin_weekday'] = np.sin(2 * np.pi * df['weekday'] / days_in_week)
#     df['cos_weekday'] = np.cos(2 * np.pi * df['weekday'] / days_in_week)
#     df['sin_monthday'] = np.sin(2 * np.pi * df['monthday'] / days_in_month)
#     df['cos_monthday'] = np.cos(2 * np.pi * df['monthday'] / days_in_month)
#     df['sin_yearday'] = np.sin(2 * np.pi * df['yearday'] / days_in_year)
#     df['cos_yearday'] = np.cos(2 * np.pi * df['yearday'] / days_in_year)
#     # drop weekday, monthday, yearday features
#     df.drop(['weekday', 'monthday', 'yearday'], inplace=True, axis=1)
#     return df


def add_ta_features(df_):
    """
    Adds technical analysis features to dataframe
    :param df_: dataframe to add technical analysis features to
    :return: dataframe with technical analysis features
    """
    # get all functions in finta
    finta_functions = [func for func in dir(TA) if callable(getattr(TA, func)) and not func.startswith("__")]
    # loop through all functions in finta and append the results to the dataframe
    # skip functions that throw errors
    for func in finta_functions:
        try:
            df_[func] = getattr(TA, func)(df_)
        except:
            pass
    return df_


def add_time_features(df_):
    # get frequency of data
    freq = pd.infer_freq(df_.index)
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
    df_['weekday'] = df_.index.dayofweek
    df_['monthday'] = df_.index.day
    df_['yearday'] = df_.index.dayofyear
    # add sine and cosine of weekday, monthday, yearday features
    df_['sin_weekday'] = np.sin(2 * np.pi * df_['weekday'] / days_in_week)
    df_['cos_weekday'] = np.cos(2 * np.pi * df_['weekday'] / days_in_week)
    df_['sin_monthday'] = np.sin(2 * np.pi * df_['monthday'] / days_in_month)
    df_['cos_monthday'] = np.cos(2 * np.pi * df_['monthday'] / days_in_month)
    df_['sin_yearday'] = np.sin(2 * np.pi * df_['yearday'] / days_in_year)
    df_['cos_yearday'] = np.cos(2 * np.pi * df_['yearday'] / days_in_year)
    # drop weekday, monthday, yearday features
    df_.drop(['weekday', 'monthday', 'yearday'], inplace=True, axis=1)
    return df_


def create_data(
    ticker='^GSPC',
    start_='2005-01-01',
    end_='2022-12-31',
    interval_='1d',
    fillna=True,
    scale_to_std=True,
    scale_min_max=False,
    fill_weekends=False,
    add_time_features_=True,
    test_split=0.2,
    shuffle=True
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
    data_df = yf.download(ticker, start_, end_, interval=interval_, progress=False)
    # rename columns
    data_df.rename(columns={"Open": "open", "Adj Close": "close", "High": "high", "Low": "low", "Volume": "volume"},
                   inplace=True)
    # drop close column
    data_df.drop("Close", inplace=True, axis=1)

    # fill weekends
    if fill_weekends:
        data_df = data_df.resample('D').ffill()

    data_df = add_ta_features(data_df)

    # drop QSTICK column
    data_df.drop("QSTICK", inplace=True, axis=1)

    # add technical analysis features
    if add_time_features_:
        data_df = add_time_features(data_df)

    # fill in missing values
    if fillna:
        data_df.fillna(method='bfill', inplace=True)
        data_df.fillna(method='ffill', inplace=True)

    train_df = data_df.iloc[:int(len(data_df) * (1 - test_split))]
    test_df = data_df.iloc[int(len(data_df) * (1 - test_split)):]

    # scale to standard deviation, by column
    if scale_to_std:
        std_ = train_df.std()
        mean_ = train_df.mean()
        train_df = (train_df - mean_) / std_
        test_df = (test_df - mean_) / std_

    # scale to min max, by column
    if scale_min_max:
        min_ = train_df.min()
        max_ = train_df.max() * 2 # we multiply by 2 to make sure we don't have any values greater than 1 in the future
        # print shape of min_ and max_
        print(min_.shape, max_.shape)
        train_df = (train_df - min_) / (max_ - min_)
        test_df = (test_df - min_) / (max_ - min_)
    if shuffle:
        train_df = train_df.sample(frac=1)
        test_df = test_df.sample(frac=1)
    if add_time_features_:
        train_df = add_time_features(train_df)
        test_df = add_time_features(test_df)
    return train_df, test_df