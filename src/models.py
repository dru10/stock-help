import os

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParametricSigmoid(nn.Module):
    def __init__(self, learnable_param=-10.0):
        super().__init__()
        self.learnable_param = learnable_param

    def forward(self, x):
        return 1 / (1 + torch.exp(self.learnable_param * x))


class BaseModel(nn.Module):
    def __init__(self, mode="linear"):
        super().__init__()
        self.net = nn.Sequential()
        self.mode = mode

    def _add_fc_layers(self, layers, dropout):
        for index in range(len(layers) - 1):
            self.net.append(
                nn.Linear(
                    in_features=layers[index], out_features=layers[index + 1]
                )
            )
            if layers[index + 1] != 1:
                self.net.append(nn.Dropout(dropout))
        # self.sigmoid = ParametricSigmoid(-100)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        if self.mode == "binary":
            x = self.sigmoid(x)
        return x

    def save_architecture(self, model_type):
        destination = os.path.join("models", self.mode, model_type)
        os.makedirs(destination, exist_ok=True)
        with open(os.path.join(destination, "summary.txt"), "w") as f:
            print(self, file=f)

    def reset_hidden(self):
        pass


class DNN(BaseModel):
    def __init__(self, input_shape, layers=[32, 32], dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        layers = [input_shape] + layers + [1]
        self._add_fc_layers(layers=layers, dropout=dropout)


class BaseRNN(BaseModel):
    def __init__(self, dropout=0.2, n_hidden=2, **kwargs):
        super().__init__(**kwargs)
        self.reccurent = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.hidden = [None for _ in range(n_hidden)]

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

    def _get_hidden(self, idx):
        hidden = self.hidden[idx]
        if hidden is None:
            return hidden

        if type(hidden) == tuple:
            return tuple([state.data for state in hidden])
        else:
            return hidden.data

    def forward(self, x):
        for idx, rec in enumerate(self.reccurent):
            x, self.hidden[idx] = rec(x, self._get_hidden(idx))
            x = self.dropout(x)
        return super().forward(x)

    def reset_hidden(self):
        self.hidden = [None for _ in self.hidden]


class LSTM(BaseRNN):
    def __init__(
        self,
        input_shape,
        hidden=[32, 32],
        layers=[16],
        num_layers=2,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__(dropout, len(hidden), **kwargs)
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
        **kwargs,
    ):
        super().__init__(dropout, len(hidden), **kwargs)
        hidden = [input_shape] + hidden
        layers = [hidden[-1]] + layers + [1]
        self._add_recurrent_layers(
            type="GRU", hidden=hidden, num_layers=num_layers, dropout=dropout
        )
        self._add_fc_layers(layers=layers, dropout=dropout)
