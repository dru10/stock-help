import os

import torch
import torch.nn as nn

from dataset import create_dataset
from eval import calculate_stats, predict_test_set
from models import DNN
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "train"
symbols = ["^SPX", "^DAX", "^BET"]
symbol = symbols[0]
model_type = "DNN1"
batch_size = 32
epochs = 100
lags = 5
epoch = 90


dataset = create_dataset(symbol)

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

model = DNN(input_shape=x_train.shape[1], layers=[32, 32]).to(device)

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

    calculate_stats(predictions, y_test, final_path)
