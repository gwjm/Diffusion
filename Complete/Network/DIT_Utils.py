import torch.nn as nn
import torch
from torch import Tensor
import math


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super(PatchEmbed, self).__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass


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
