import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_shape, layers=[32, 32]):
        super().__init__()
        layers = [input_shape] + layers + [1]
        self.net = nn.Sequential()
        for index in range(len(layers) - 1):
            self.net.append(
                nn.Linear(
                    in_features=layers[index], out_features=layers[index + 1]
                )
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        x = self.sigmoid(x)
        return x
