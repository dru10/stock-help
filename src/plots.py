import os

import matplotlib.pyplot as plt


def loss_curves(train_loss, val_loss, model_path):
    plt.figure()
    plt.plot(train_loss.values(), label="Training Loss")
    plt.plot(val_loss.values(), label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Model training loss curves")
    plt.savefig(os.path.join(model_path, "loss_curves.jpg"))
