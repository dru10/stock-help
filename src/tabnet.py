import os
from collections import defaultdict

from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

import eval
import plots
from dataset import create_dataset


class ModelCheckpoint(Callback):
    def __init__(self, model, model_path, checkpoint_interval=10):
        self.checkpoint_interval = checkpoint_interval
        self.model = model
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        if (
            epoch % self.checkpoint_interval == 0
            or epoch == self.model.max_epochs - 1
        ):
            final_path = os.path.join(self.model_path, str(epoch))
            os.makedirs(final_path, exist_ok=True)

            self.model.save_model(os.path.join(final_path, "params"))


mode = "eval"
symbols = ["^SPX", "^DAX", "^BET"]
epochs = 100
ds_mode = "price"

for symbol in symbols:
    ds = create_dataset(symbol, mode=ds_mode)
    lag_predictions = defaultdict(dict)

    x_train, y_train = ds["train"]["X"], ds["train"]["Y"]
    x_valid, y_valid = ds["valid"]["X"], ds["valid"]["Y"]
    x_test, y_test = ds["test"]["X"], ds["test"]["Y"]

    if ds_mode == "price":
        y_train = y_train.reshape(-1, 1)
        y_valid = y_valid.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    for batch_size in [32, 1024]:
        for model_type in ["TNN1", "TNN2", "TNN3"]:
            for lags in [5, 10, 20]:
                tabnet_kwargs = {
                    "TNN1": {"n_d": 8, "n_a": 8, "n_steps": 3},
                    "TNN2": {"n_d": 16, "n_a": 16, "n_steps": 3},
                    "TNN3": {"n_d": 32, "n_a": 32, "n_steps": 3},
                }

                model_mode = "linear" if ds_mode == "price" else "binary"
                model = (
                    TabNetRegressor(**tabnet_kwargs[model_type])
                    if model_mode == "linear"
                    else TabNetClassifier(**tabnet_kwargs[model_type])
                )
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
                os.makedirs(model_path, exist_ok=True)

                if mode == "train":
                    save_every_epochs = ModelCheckpoint(
                        model=model,
                        model_path=model_path,
                        checkpoint_interval=10,
                    )

                    eval_metric = "mse" if model_mode == "linear" else "logloss"

                    model.fit(
                        x_train[:, :lags],
                        y_train,
                        eval_set=[(x_valid[:, :lags], y_valid)],
                        eval_metric=[eval_metric],
                        max_epochs=100,
                        patience=0,
                        batch_size=batch_size,
                        callbacks=[save_every_epochs],
                    )

                    plots.loss_curves(
                        model.history["loss"],
                        model.history[f"val_0_{eval_metric}"],
                        model_path,
                    )
                elif mode == "eval":
                    for epoch in list(range(0, epochs, 10)) + [epochs - 1]:
                        final_path = os.path.join(model_path, str(epoch))
                        model.load_model(os.path.join(final_path, "params.zip"))

                        predictions = model.predict(x_test[:, :lags])
                        lag_predictions[epoch][lags] = predictions

                        if model_mode == "linear":
                            eval.calculate_linear_stats(
                                predictions, y_test, final_path
                            )
                        elif model_mode == "binary":
                            eval.calculate_stats(
                                predictions, y_test, final_path
                            )

                        dates = ds["test"]["dates"]
                        if ds_mode == "price":
                            scaler = ds["scaler"]
                            eval.evaluate_linear_predictions(
                                symbol=symbol,
                                model_path=final_path,
                                pred=predictions,
                                true=y_test,
                                scaler=scaler,
                                dates=dates,
                            )
                        elif ds_mode == "logs":
                            log_rets = ds["test"]["log_ret"]
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
