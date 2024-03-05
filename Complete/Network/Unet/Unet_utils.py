import torch

import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, hidden_channels=None):
        super().__init__()
        if not hidden_channels:
            hidden_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(hidden_channels),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_channels), nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, t):
        x = self.conv1(x)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2]
        x += time_emb
        x = self.conv2(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, hidden_channels=None):
        super().__init__()
        if not hidden_channels:
            hidden_channels = out_channels

        self.down_sample = nn.Sequential(
            DoubleConv(in_channels, out_channels, time_emb_dim, hidden_channels),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

    def forward(self, x, t):
        return self.down_sample(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, bilinear=True):
        super().__init__()

        if bilinear:
            self.up_sample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up_sample = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, time_emb_dim, out_channels)

    def forward(self, x_in, x_res, t):
        x = self.up_sample(x_in)

        diffY = x.shape[2] - x_res.shape[2]
        diffX = x.shape[3] - x_res.shape[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x_res, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
