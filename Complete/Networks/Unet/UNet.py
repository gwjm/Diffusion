import torch.nn as nn
from UNet_utils import * 


class U_Net(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, bilinear:bool = True, hidden_dims: list = None) -> None:
        if hidden_dims: 
            assert len(hidden_dims) % 2 == 0
        else: 
            factor = 2 if bilinear else 1 
            hidden_dims = [64, 
                           128, 
                           256, 
                           512, 
                           1024 // factor, 
                           512 // factor, 
                           256 // factor, 
                           128 // factor, 
                           64]
        self.n_channels = in_channels
        self.n_classes = num_classes 
        self.bilinear = bilinear
        
        modules = []
        modules.append(DoubleConv(in_channels, hidden_dims[0]))

        for i in range(0, len(hidden_dims) // 2):
            modules.append(DownConv(hidden_dims[i], hidden_dims[i+1]))
        
        for i in range(len(hidden_dims)//2, len(hidden_dims)):
            modules.append(UpConv(hidden_dims[i], hidden_dims[i+1], bilinear))
        
        modules.append(OutConv(hidden_dims[-1], num_classes))

        self.Model = nn.Sequential(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Model(x)
