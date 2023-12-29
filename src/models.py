import os

import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential()

    def _add_fc_layers(self, layers, dropout):
        for index in range(len(layers) - 1):
            self.net.append(
                nn.Linear(
                    in_features=layers[index], out_features=layers[index + 1]
                )
            )
            if layers[index + 1] != 1:
                self.net.append(nn.Dropout(dropout))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        x = self.sigmoid(x)
        return x

    def save_architecture(self, model_type):
        destination = os.path.join("models", model_type)
        os.makedirs(destination, exist_ok=True)
        with open(os.path.join(destination, "summary.txt"), "w") as f:
            print(self, file=f)

    def reset_hidden(self):
        pass


class DNN(BaseModel):
    def __init__(self, input_shape, layers=[32, 32], dropout=0.2):
        super().__init__()
        layers = [input_shape] + layers + [1]
        self._add_fc_layers(layers=layers, dropout=dropout)


class BaseRNN(BaseModel):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.reccurent = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.hidden = None

    def _add_recurrent_layers(self, type, hidden, num_layers, dropout):
        for idx in range(len(hidden) - 1):
            self.reccurent.append(
                getattr(nn, type)(
                    input_size=hidden[idx],
                    hidden_size=hidden[idx + 1],
                    num_layers=num_layers,
                    dropout=dropout,
                )
            )

    def _get_hidden(self):
        if self.hidden is None:
            return self.hidden

        if type(self.hidden) == tuple:
            return tuple([state.data for state in self.hidden])
        else:
            return self.hidden.data

    def forward(self, x):
        for rec in self.reccurent:
            x, self.hidden = rec(x, self._get_hidden())
            x = self.dropout(x)
            return super().forward(x)

    def reset_hidden(self):
        self.hidden = None


class LSTM(BaseRNN):
    def __init__(
        self,
        input_shape,
        hidden=[32, 32],
        layers=[16],
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__(dropout)
        hidden = [input_shape] + hidden
        layers = [hidden[-1]] + layers + [1]
        self._add_recurrent_layers(
            type="LSTM", hidden=hidden, num_layers=num_layers, dropout=dropout
        )
        self._add_fc_layers(layers=layers, dropout=dropout)


class GRU(BaseRNN):
    def __init__(
        self,
        input_shape,
        hidden=[32, 32],
        layers=[16],
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        hidden = [input_shape] + hidden
        layers = [hidden[-1]] + layers + [1]
        self._add_recurrent_layers(
            type="GRU", hidden=hidden, num_layers=num_layers, dropout=dropout
        )
        self._add_fc_layers(layers=layers, dropout=dropout)
