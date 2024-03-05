import torch
import torch.nn as nn
import math

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


def extract_values(x, t):
    batch_size = t.shape[0]
    out = x.gather(-1, t.cpu())
    return out.reshape(batch_size, 1, 1, 1).to(t.device)


class DiffusionForwardProcess(nn.Module):
    def __init__(self, Betas):
        super(DiffusionForwardProcess, self).__init__()
        self.Betas = Betas
        self.alphas = 1.0 - Betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @torch.no_grad()
    def forward(self, x_0, t):
        B, C, H, W = x_0.shape
        noise = torch.randn((B, C, H, W)).to(t.device)
        x_t = (
            extract_values(self.sqrt_alphas_cumprod, t) * x_0
            + extract_values(self.sqrt_1m_alphas_cumprod, t) * noise
        )
        return x_t, noise


class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.embedding_table = nn.Embedding(num_patches, embedding_dim)

    def forward(self, x):
        _, T, _ = x.shape
        return x + self.embedding_table(torch.arange(T))


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
