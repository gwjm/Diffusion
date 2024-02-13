import torch.nn as nn


class U_Net(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, bilinear:bool = True) -> None:
        self.n_channels = in_channels
        self.n_classes = num_classes 
        self.bilinear = bilinear

    def forward(self, x):
        pass
