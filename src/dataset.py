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


def create_dataset(symbol, valid_start="2022-01-01", valid_end="2023-01-01"):
    df = web.DataReader(
        symbol, "stooq", start="2011-01-01", end="2023-12-31"
    ).iloc[::-1]

    df = df.filter(["Close"])
    df.dropna(inplace=True)

    log_rets = np.log(df / df.shift(1)).dropna()

    lagged, cols = add_lags(log_rets)

    binaries = np.digitize(lagged, bins=[0])

    log_rets_close = np.array(lagged["Close"])

    n_train = len(lagged.loc[:valid_start])
    n_valid = len(lagged.loc[valid_start:valid_end])

    return {
        "train": {
            "X": binaries[:n_train, 1:],
            "Y": binaries[:n_train, 0],
            "log_ret": log_rets_close[:n_train],
        },
        "valid": {
            "X": binaries[n_train : n_train + n_valid, 1:],
            "Y": binaries[n_train : n_train + n_valid, 0],
            "log_ret": log_rets_close[n_train : n_train + n_valid],
        },
        "test": {
            "X": binaries[n_train + n_valid :, 1:],
            "Y": binaries[n_train + n_valid :, 0],
            "log_ret": log_rets_close[n_train + n_valid :],
        },
    }
