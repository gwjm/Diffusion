import torch
import torch.nn as nn

"""
Beta Schedule Generators
"""


def linear_beta_schedule(
    self, beta_start: torch.float, beta_end: torch.float, timesteps: int
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(
    self, beta_start: torch.float, beta_end: torch.float, timesteps: int
) -> torch.Tensor:
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(
    self, beta_start: torch.float, beta_end: torch.float, timesteps: int
) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def cos_beta_schedule(
    self,
    beta_start: torch.float,
    beta_end: torch.float,
    timesteps: int,
    s: torch.float = 0.008,
) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, timesteps)
    return torch.cos(betas * torch.pi / 2.0) ** 2


"""
Sinusoidal Positional Embeddings for time steps in the UNET

"""


class PositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
