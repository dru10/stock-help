import os
from collections import defaultdict

import torch
import torch.nn as nn

import eval
import models
from dataset import create_dataset
from train import train

mode = "eval"
symbols = ["^SPX", "^DAX", "^BET"]
epochs = 100
ds_mode = "price"

for symbol in symbols:
    dataset = create_dataset(symbol, mode=ds_mode)

    lag_predictions = defaultdict(dict)

    for batch_size in [32, 1024]:
        for model_type in [
            "DNN1",
            "DNN2",
            "DNN3",
            "LSTM1",
            "LSTM2",
            "LSTM3",
            "GRU1",
            "GRU2",
            "GRU3",
        ]:
            for lags in [5, 10, 20]:
                x_train, y_train = dataset["train"]["X"], dataset["train"]["Y"]
                x_valid, y_valid = dataset["valid"]["X"], dataset["valid"]["Y"]
                x_test, y_test = dataset["test"]["X"], dataset["test"]["Y"]

                # Convert to tensors
                x_train = torch.tensor(x_train[:, :lags], dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(
                    1
                )
                x_valid = torch.tensor(x_valid[:, :lags], dtype=torch.float32)
                y_valid = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(
                    1
                )
                x_test = torch.tensor(x_test[:, :lags], dtype=torch.float32)
                y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

                models_kwargs = {
                    "DNN1": {
                        "input_shape": x_train.shape[1],
                        "layers": [32, 32],
                    },
                    "DNN2": {
                        "input_shape": x_train.shape[1],
                        "layers": [64, 64],
                    },
                    "DNN3": {
                        "input_shape": x_train.shape[1],
                        "layers": [64, 32],
                    },
                    "LSTM1": {"input_shape": x_train.shape[1]},
                    "LSTM2": {
                        "input_shape": x_train.shape[1],
                        "hidden": [50],
                        "layers": [25],
                        "num_layers": 2,
                    },
                    "LSTM3": {
                        "input_shape": x_train.shape[1],
                        "hidden": [64, 32],
                        "layers": [32, 16],
                        "num_layers": 4,
                        "dropout": 0.4,
                    },
                    "GRU1": {"input_shape": x_train.shape[1]},
                    "GRU2": {
                        "input_shape": x_train.shape[1],
                        "hidden": [50],
                        "layers": [25],
                        "num_layers": 2,
                    },
                    "GRU3": {
                        "input_shape": x_train.shape[1],
                        "hidden": [64, 32],
                        "layers": [32, 16],
                        "num_layers": 4,
                        "dropout": 0.4,
                    },
                }
                model_mode = "linear" if ds_mode == "price" else "binary"
                model = getattr(models, model_type[:-1])(
                    **models_kwargs[model_type], mode=model_mode
                )

                if mode == "train":
                    model.save_architecture(model_type)

                if model_mode == "linear":
                    criterion = nn.MSELoss()
                elif model_mode == "binary":
                    # Binary Cross Entropy Loss
                    criterion = nn.BCELoss()

                # Adam optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                model_path = os.path.join(
                    "models",
                    model_mode,
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
                    for epoch in list(range(0, epochs, 10)) + [epochs - 1]:
                        final_path = os.path.join(model_path, str(epoch))
                        model.load_state_dict(
                            torch.load(os.path.join(final_path, "params.pt"))
                        )

                        predictions = eval.predict_test_set(
                            model, x_test, y_test, mode=model_mode
                        )

                        lag_predictions[epoch][lags] = predictions

                        if model_mode == "linear":
                            eval.calculate_linear_stats(
                                predictions, y_test, final_path
                            )
                        elif model_mode == "binary":
                            eval.calculate_stats(
                                predictions, y_test, final_path
                            )

                        dates = dataset["test"]["dates"]
                        if ds_mode == "price":
                            scaler = dataset["scaler"]
                            eval.evaluate_linear_predictions(
                                symbol=symbol,
                                model_path=final_path,
                                pred=predictions,
                                true=y_test,
                                scaler=scaler,
                                dates=dates,
                            )
                        elif ds_mode == "logs":
                            log_rets = dataset["test"]["log_ret"]
                            eval.evaluate_predictions(
                                symbol=symbol,
                                model_path=final_path,
                                pred=predictions,
                                true=y_test,
                                log_close=log_rets,
                                dates=dates,
                            )

            if mode == "eval":
                for epoch in list(range(0, epochs, 10)) + [epochs - 1]:
                    if ds_mode == "price":
                        eval.evaluate_all_lags_price(
                            dates,
                            y_test,
                            lag_predictions[epoch],
                            scaler,
                            symbol,
                            model_type,
                            batch_size,
                            epoch,
                        )
                    elif ds_mode == "logs":
                        eval.evaluate_all_lags(
                            dates,
                            log_rets,
                            lag_predictions[epoch],
                            symbol,
                            model_type,
                            batch_size,
                            epoch,
                        )
                print(
                    f"Finished {model_type} for symbol {symbol} with batch size {batch_size}"
                )
