import os

from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_model import TabNetClassifier

import plots
from dataset import create_dataset
from eval import calculate_stats, evaluate_all_lags, evaluate_predictions


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


mode = "train"
symbols = ["^SPX", "^DAX", "^BET"]
symbol = symbols[0]
model_type = "TNN3"
epochs = 100
batch_size = 32
lags = 5
epoch = 20

ds = create_dataset(symbol)

x_train, y_train = ds["train"]["X"], ds["train"]["Y"]
x_valid, y_valid = ds["valid"]["X"], ds["valid"]["Y"]
x_test, y_test = ds["test"]["X"], ds["test"]["Y"]

tabnet_kwargs = {
    "TNN1": {"n_d": 8, "n_a": 8, "n_steps": 3},
    "TNN2": {"n_d": 16, "n_a": 16, "n_steps": 3},
    "TNN3": {"n_d": 32, "n_a": 32, "n_steps": 3},
}

model = TabNetClassifier(**tabnet_kwargs[model_type])

lag_predictions = {}

for lags in [5, 10, 20]:
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
    os.makedirs(model_path, exist_ok=True)

    if mode == "train":
        save_every_epochs = ModelCheckpoint(
            model=model, model_path=model_path, checkpoint_interval=10
        )

        model.fit(
            x_train[:, :lags],
            y_train,
            eval_set=[(x_valid[:, :lags], y_valid)],
            eval_metric=["logloss"],
            max_epochs=100,
            patience=0,
            batch_size=batch_size,
            callbacks=[save_every_epochs],
        )

        plots.loss_curves(
            model.history["loss"], model.history["val_0_logloss"], model_path
        )
    elif mode == "eval":
        final_path = os.path.join(model_path, str(epoch))
        model.load_model(os.path.join(final_path, "params.zip"))

        predictions = model.predict(x_test[:, :lags])

        lag_predictions[lags] = predictions

        calculate_stats(predictions, y_test, final_path)

        log_rets = ds["test"]["log_ret"]
        dates = ds["test"]["dates"]

        evaluate_predictions(
            symbol=symbol,
            model_path=final_path,
            pred=predictions,
            true=y_test,
            log_close=log_rets,
            dates=dates,
        )

if mode == "eval":
    evaluate_all_lags(
        dates, log_rets, lag_predictions, symbol, model_type, batch_size, epoch
    )
