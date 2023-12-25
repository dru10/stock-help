import torch
import torch.nn as nn

from dataset import create_dataset
from eval import evaluate_model
from models import DNN
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

symbols = ["^SPX", "^DAX", "^BET"]
symbol = symbols[0]


dataset = create_dataset(symbol)

x_train, y_train = dataset["train"]["X"], dataset["train"]["Y"]
x_valid, y_valid = dataset["valid"]["X"], dataset["valid"]["Y"]
x_test, y_test = dataset["test"]["X"], dataset["test"]["Y"]

# Convert to tensors
x_train = torch.tensor(x_train[:, :5], dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_valid = torch.tensor(x_valid[:, :5], dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)
x_test = torch.tensor(x_test[:, :5], dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

model = DNN(input_shape=x_train.shape[1], layers=[32, 32]).to(device)

# Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loss, valid_loss, model_path = train(
    model=model,
    type="DNN1",
    symbol=symbol,
    train_valid=(x_train, y_train, x_valid, y_valid),
    criterion=criterion,
    optimizer=optimizer,
    epochs=1,
    batch_size=32,
)

evaluate_model(model, x_test, y_test, model_path)
