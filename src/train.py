import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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


def train(model, train_valid, criterion, optimizer, epochs, batch_size=1):
    """
    Generic training loop

    model
    train_valid: (x_train, y_train, x_valid, y_valid)
    criterion
    optimizer
    epochs
    batch_size: default = 1 means one example represents one batch
    """
    model = model.to(device)
    x_train, y_train, x_valid, y_valid = train_valid

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
                mode="eval",
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
