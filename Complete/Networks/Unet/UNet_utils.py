import torch
import torch.nn as nn 
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = None) -> None:
        super().__init__()
        if not hidden_channels: 
            hidden_channels = out_channels
        self.conv = nn.Sequential(
         nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
         nn.BatchNorm2d(hidden_channels),
         nn.ReLU6(inplace=True), # NOTE: Experimenting with the RELU6 activation func for fun
         nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
         nn.BatchNorm2d(out_channels),
         nn.ReLU6(inplace=True)
         )

    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.conv(x)

class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = None) -> None:
        super().__init__()
        if not hidden_channels:
            hidden_channels = out_channels

        self.down_sample = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                  DoubleConv(in_channels, out_channels, hidden_channels))

    def forward(self, x):
        return self.down_sample(x)
    
class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()

        if bilinear: 
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else: 
            self.up_sample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_in: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:
        x = self.up_sample(x_in)

        diffY = x.shape[2] - x_res.shape[2]
        diffX = x.shape[3] - x_res.shape[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x_res, x], dim = 1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int)-> None:
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.conv(x)