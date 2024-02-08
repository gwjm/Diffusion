import torch.nn as nn
import torch
import torch.nn.functional as F
import math

"""
The goal of this is to understand how a diffusion encoder works

Assumptions: 
For Simplicity, I'm going to only use one beta value, and assume a standard normal distribution for each timestep with mean of 0 and sd of 1. 
"""


class SimpleDiffusionEncoder(nn.Module):
    def __init__(self, Beta: torch.float, max_timesteps: int) -> None:
        super().__init__()
        self.beta = Beta

    def forward(self, x: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        Params:
        x: torch.Tensor: Input image
        timestep: int: number of timesteps to step forward in the noising process
        """
        if timestep == 0:
            return x

        x *= torch.sqrt(torch.full((1, 1), 1.0 - self.beta))
        noise = torch.sqrt(torch.full((1, 1), self.beta)) * (
            torch.normal(0, 1, size=[x.shape[0], x.shape[1], x.shape[2]]) / 256.0
        )
        x += noise

        timestep -= 1
        if timestep == 0:
            return x
        else:
            return self.forward(x, timestep)


"""
Below is a more complete diffusion encoding module, with absolute credit for the methods to huggingface's excellent annotated Diffusion model blog post 
https://huggingface.co/blog/annotated-diffusion
"""


class Q_Sampler(nn.Module):
    def __init__(
        self, beta_schedule: torch.tensor, max_timesteps: torch.tensor
    ) -> None:
        super(
            Q_Sampler,
        )
        self.beta_schedule = beta_schedule
        self.max_timesteps = max_timesteps
        self.alpha, self.alpha_complement = self.calculate_alphas()

    def calculate_alphas(self) -> torch.Tensor:
        """
        This function will calculate the alphas for each timestep
        both on the initialization of the encoder object and any time
        the beta schedule is changed.
        """
        alphas = 1 - self.beta_schedule
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_cumprod_compliment = torch.sqrt(1.0 - alphas_cumprod)
        return sqrt_cumprod, sqrt_cumprod_compliment

    def forward(self, x: torch.tensor, timestep: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = self.extract_alpha(self.alpha, timestep, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract_alpha(
            self.alpha_complement, timestep, x.shape
        )

        return (
            sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise,
            noise,
        )

    def extract_alpha(
        self, alpha: torch.Tensor, timestep: torch.Tensor, x_shape: int
    ) -> torch.Tensor:
        batch_size = timestep.shape[0]
        out = alpha.gather(-1, timestep.cpu().long())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(timestep.device)
