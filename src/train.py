import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import plots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perform_one_epoch(mode, model, loader, criterion, optimizer):
    loss_per_batch = []
    for input, label in loader:
        X = input.to(device)
        y_true = label.to(device)

        # Reset hidden state for RNNs

        if mode == "train":
            optimizer.zero_grad()

        y_pred = model(X)

        loss = criterion(y_pred, y_true)

        if mode == "train":
            loss.backward()
            optimizer.step()

        loss_per_batch.append(loss.item())
    return loss_per_batch


def save_model_checkpoint(model, model_path, epoch):
    final_path = os.path.join(model_path, str(epoch))
    os.makedirs(final_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(final_path, "params.pt"))


def train(
    model,
    model_path,
    train_valid,
    criterion,
    optimizer,
    epochs,
    batch_size=1,
    checkpoint_interval=10,
):
    """
    Generic training loop

    model
    type: name of the model to be saved
    symbol
    train_valid: (x_train, y_train, x_valid, y_valid)
    criterion
    optimizer
    epochs
    batch_size: default = 1 means one example represents one batch
    checkpoint_interval: default = 10 after how many epochs to save a checkpoint
    """
    model = model.to(device)
    x_train, y_train, x_valid, y_valid = train_valid

    os.makedirs(model_path, exist_ok=True)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False
    )
    valid_loader = DataLoader(
        TensorDataset(x_valid, y_valid), batch_size=batch_size, shuffle=False
    )

    train_loss = defaultdict(float)
    valid_loss = defaultdict(float)

    for epoch in range(epochs):
        epoch_start = time.time()

        # init hidden state for rnns

        model.train()

        epoch_train_loss = perform_one_epoch(
            mode="train",
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )
        train_loss[epoch] = np.array(epoch_train_loss).mean()

        # Need to evaluate on validation set
        model.eval()
        with torch.no_grad():
            epoch_valid_loss = perform_one_epoch(
                mode="valid",
                model=model,
                loader=valid_loader,
                criterion=criterion,
                optimizer=optimizer,
            )
            valid_loss[epoch] = np.array(epoch_valid_loss).mean()

        epoch_end = time.time()

        print(
            f"Epoch: [{epoch:{len(str(epochs))}}/{epochs}]",
            f"Train Loss: {train_loss[epoch]:3.8f}",
            f"Valid Loss: {valid_loss[epoch]:3.8f}",
            f"Time: {epoch_end - epoch_start:.2f} s",
        )

        # Save model checkpoints
        if epoch % checkpoint_interval == 0 or epoch == epochs - 1:
            save_model_checkpoint(model, model_path, epoch)

            plots.loss_curves(train_loss, valid_loss, model_path)

    return train_loss, valid_loss
