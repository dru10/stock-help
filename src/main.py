import torch
import torch.nn as nn

from dataset import create_dataset
from models import DNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

symbols = ["^SPX", "^DAX", "^BET"]
symbol = symbols[0]


dataset = create_dataset(symbol)

x_train, y_train = dataset["train"]["X"], dataset["train"]["Y"]

# Convert to tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

model = DNN(input_shape=x_train.shape[1], layers=[8, 8]).to(device)

# Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
