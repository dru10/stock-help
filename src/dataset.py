import numpy as np
import pandas_datareader.data as web


def add_lags(df, lags=20):
    cols = []
    new_df = df.copy(deep=True)
    for lag in range(1, lags + 1):
        col = f"lag_{lag}"
        new_df[col] = df.shift(lag)
        cols.append(col)
    new_df.dropna(inplace=True)
    return new_df, cols


def create_dataset(symbol):
    df = web.DataReader(
        symbol, "stooq", start="2011-01-01", end="2023-12-31"
    ).iloc[::-1]

    df = df.filter(["Close"])
    df.dropna(inplace=True)

    log_rets = np.log(df / df.shift(1)).dropna()

    return add_lags(log_rets)
