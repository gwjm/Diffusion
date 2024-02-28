import torch.nn as nn
import torch
from torch import Tensor
import math


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embedding_dim: int,
        bias=True,
    ) -> None:
        super(PatchEmbed, self).__init__()
        img_shape = (img_size, img_size)
        patch_shape = (patch_size, patch_size)
        self.img_shape = img_shape
        self.patch_shape = patch_shape
        self.grid_shape = (
            img_shape[0] // patch_shape[0],
            img_shape[1] // patch_shape[1],
        )
        self.num_patches = self.grid_shape[0] * self.grid_shape[1]
        self.proj = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, freq_embedding_dim: int) -> None:
        super(TimestepEmbedder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embedding_dim = freq_embedding_dim

    @staticmethod
    def get_timestep_embedding(
        self, t: Tensor, out_dim: int, max_period: int
    ) -> Tensor:
        half = out_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if out_dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, x: Tensor) -> Tensor:
        freq_embedding = self.get_timestep_embedding(x, self.freq_embedding_dim)
        return self.mlp(freq_embedding)


class LabelEmbedder(nn.Module):
    def __init__(self, n_classes: int, embedding_dim) -> None:
        super(LabelEmbedder, self).__init__()
        self.embedding_table = nn.Embedding(n_classes, embedding_dim)
        self.n_classes = n_classes

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding_table(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.embedding_table = nn.Embedding(num_patches, embedding_dim)

    def forward(self, x):
        _, T, _ = x.shape
        return x + self.embedding_table(torch.arange(T))
