import torch.nn as nn
from Unet_utils import *
from utils import SinusoidalPositionalEmbedding


class U_Net(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        time_emb_dim,
        bilinear=True,
        hidden_dims=None,
    ):
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.Relu(),
        )
        if hidden_dims:
            assert len(hidden_dims) % 2 == 0
        else:
            factor = 2 if bilinear else 1
            hidden_dims = [
                64,
                128,
                256,
                512,
                1024 // factor,
                512 // factor,
                256 // factor,
                128 // factor,
                64,
            ]
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.bilinear = bilinear

        self.conv1 = DoubleConv(in_channels, time_emb_dim, hidden_dims[0])

        down_modules = []
        for i in range(0, len(hidden_dims) // 2):
            down_modules.append(
                DownConv(hidden_dims[i], hidden_dims[i + 1], time_emb_dim)
            )

        up_modules = []
        for i in range(len(hidden_dims) // 2, len(hidden_dims)):
            up_modules.append(
                UpConv(hidden_dims[i], hidden_dims[i + 1], time_emb_dim, bilinear)
            )

        self.Out = OutConv(hidden_dims[-1], num_classes)

    def forward(self, x, timestep) -> torch.Tensor:
        # TODO: Implement the forward pass with residual connections and timestep embeddings
        pass
