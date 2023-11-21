from torch import nn


class MyComponent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1/x
