import os

import matplotlib.pyplot as plt


def loss_curves(train_loss, val_loss, model_path):
    try:
        train_loss = train_loss.values()
        val_loss = val_loss.values()
    except AttributeError:
        pass

    plt.figure()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Model training loss curves")
    plt.savefig(os.path.join(model_path, "loss_curves.png"))
    plt.close()


def returns(dates, real, candidates, symbol, plot_destination, labels):
    plt.style.use("fivethirtyeight")

    plt.figure(figsize=(16, 8))
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.plot(dates, real)
    for candidate in candidates:
        plt.plot(dates, candidate)
    plt.legend(labels)
    plt.title(f"Returns Simulation for {symbol}")
    plt.savefig(plot_destination)
    plt.close()


def price_predictions(dates, pred, true, symbol, plot_destination, labels):
    plt.style.use("fivethirtyeight")

    plt.figure(figsize=(16, 8))
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.plot(dates, pred)
    plt.plot(dates, true)
    plt.legend(labels)
    plt.title(f"Price Predictions for {symbol}")
    plt.savefig(plot_destination)
    plt.close()
