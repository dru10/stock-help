import torch
import torch.nn as nn

from dataset import create_dataset
from models import DNN
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

symbols = ["^SPX", "^DAX", "^BET"]
symbol = symbols[0]


dataset = create_dataset(symbol)

x_train, y_train = dataset["train"]["X"], dataset["train"]["Y"]
x_valid, y_valid = dataset["valid"]["X"], dataset["valid"]["Y"]

# Convert to tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_valid = torch.tensor(x_valid, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)

model = DNN(input_shape=x_train.shape[1], layers=[8, 8]).to(device)

# Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(
    model=model,
    train_valid=(x_train, y_train, x_valid, y_valid),
    criterion=criterion,
    optimizer=optimizer,
    epochs=100,
    batch_size=32,
)
