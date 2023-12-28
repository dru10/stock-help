import os

import numpy as np
import torch
import torch.nn as nn

import plots
from dataset import create_dataset
from eval import (
    calculate_return,
    calculate_stats,
    evaluate_predictions,
    predict_test_set,
    random_returns,
)
from models import DNN
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "eval"
symbols = ["^SPX", "^DAX", "^BET"]
symbol = symbols[0]
model_type = "DNN2"
batch_size = 64
epochs = 100
lags = 10
epoch = 99


dataset = create_dataset(symbol)

lag_predictions = {}

for lags in [5, 10, 20]:
    x_train, y_train = dataset["train"]["X"], dataset["train"]["Y"]
    x_valid, y_valid = dataset["valid"]["X"], dataset["valid"]["Y"]
    x_test, y_test = dataset["test"]["X"], dataset["test"]["Y"]

    # Convert to tensors
    x_train = torch.tensor(x_train[:, :lags], dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_valid = torch.tensor(x_valid[:, :lags], dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)
    x_test = torch.tensor(x_test[:, :lags], dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    model = DNN(input_shape=x_train.shape[1], layers=[64, 64]).to(device)
    if mode == "train":
        model.save_architecture(model_type)

    # Binary Cross Entropy Loss
    criterion = nn.BCELoss()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model_path = os.path.join(
        "models",
        model_type,
        symbol,
        "lags",
        str(lags),
        "batch_size",
        str(batch_size),
        "epochs",
    )

    if mode == "train":
        train_loss, valid_loss = train(
            model=model,
            model_path=model_path,
            train_valid=(x_train, y_train, x_valid, y_valid),
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
        )
    elif mode == "eval":
        # for epoch in list(range(0, 100, 10)) + [epochs - 1]:
        final_path = os.path.join(model_path, str(epoch))
        model.load_state_dict(torch.load(os.path.join(final_path, "params.pt")))

        predictions = predict_test_set(model, x_test, y_test)

        lag_predictions[lags] = predictions

        calculate_stats(predictions, y_test, final_path)

        log_rets = dataset["test"]["log_ret"]
        dates = dataset["test"]["dates"]

        evaluate_predictions(
            symbol=symbol,
            model_path=final_path,
            pred=predictions,
            true=y_test,
            log_close=log_rets,
            dates=dates,
        )

if mode == "eval":
    lag_rets = {
        lag: calculate_return(pred, log_rets)
        for lag, pred in lag_predictions.items()
    }
    real_return = np.exp(np.cumsum(log_rets))
    random_return = random_returns(log_rets)

    plots.returns(
        dates,
        real_return,
        candidates=[lag_rets[key] for key in lag_rets] + [random_return],
        symbol=symbol,
        plot_destination=os.path.join(
            "models",
            model_type,
            symbol,
            "lags",
            f"all_returns_batch_size_{batch_size}_epoch_{epoch}.png",
        ),
        labels=["Real"] + [f"{key} lags" for key in lag_rets] + ["Random"],
    )
