import torch

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
