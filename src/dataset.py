import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.preprocessing import MinMaxScaler


def add_lags(df, lags=20):
    cols = []
    new_df = df.copy(deep=True)
    for lag in range(1, lags + 1):
        col = f"lag_{lag}"
        new_df[col] = df.shift(lag)
        cols.append(col)
    new_df.dropna(inplace=True)
    return new_df, cols


def fetch_data(symbol, start="2011-01-01", end="2023-12-31"):
    df = web.DataReader(symbol, "stooq", start=start, end=end).iloc[::-1]
    df = df.filter(["Close"])
    df.dropna(inplace=True)
    return df


def scale_prices(df, train_split: str):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = df.loc[:train_split]
    scaler.fit(train)
    scaled = scaler.transform(df)
    # Convert back to DataFrame
    scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return scaled, scaler


def create_dataset(
    symbol, valid_start="2022-01-01", valid_end="2023-01-01", mode="price"
):
    df = fetch_data(symbol)

    if mode == "price":
        df_scaled, scaler = scale_prices(df, train_split="2022-01-01")
    elif mode == "logs":
        df_scaled = np.log(df / df.shift(1)).dropna()

    lagged, cols = add_lags(df_scaled)
    n_train = len(lagged.loc[:valid_start])
    n_valid = len(lagged.loc[valid_start:valid_end])
    dates = lagged[n_train + n_valid :].index

    if mode == "price":
        lagged = np.array(lagged)
    elif mode == "logs":
        log_rets_close = np.array(lagged["Close"])
        lagged = np.digitize(lagged, bins=[0])

    payload = {
        "train": {
            "X": lagged[:n_train, 1:],
            "Y": lagged[:n_train, 0],
        },
        "valid": {
            "X": lagged[n_train : n_train + n_valid, 1:],
            "Y": lagged[n_train : n_train + n_valid, 0],
        },
        "test": {
            "X": lagged[n_train + n_valid :, 1:],
            "Y": lagged[n_train + n_valid :, 0],
            "dates": dates,
        },
    }

    if mode == "price":
        payload["scaler"] = scaler
    elif mode == "logs":
        payload["train"]["log_ret"] = log_rets_close[:n_train]
        payload["valid"]["log_ret"] = log_rets_close[
            n_train : n_train + n_valid
        ]
        payload["test"]["log_ret"] = log_rets_close[n_train + n_valid :]

    return payload
