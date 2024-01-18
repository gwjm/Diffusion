import torch.nn as nn
import torch
import numpy as np

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


class DiffusionEncoder(nn.Module):
    def __init__(self, beta_schedule: torch.tensor, max_timesteps: int) -> None:
        super(
            DiffusionEncoder,
        )
        self.schedule = beta_schedule
        self.max_timesteps = max_timesteps

    def forward(self, x: torch.tensor, timestep: int) -> torch.Tensor:
        pass

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
        pass

    def set_beta_schedule(
        self,
        timesteps: int,
        beta_start: torch.float = 0.0001,
        beta_end: torch.float = 0.02,
        schedule_type: str = "linear",
    ) -> None:
        self.max_timesteps = timesteps
        schedule_type = schedule_type.lower()
        if schedule_type is None or schedule_type == "linear":
            self.schedule = self.linear_beta_schedule(beta_start, beta_end, timesteps)
        elif schedule_type == "sigmoid":
            self.schedule = self.sigmoid_beta_schedule(beta_start, beta_end, timesteps)
        elif schedule_type == "quadratic":
            self.schedule = self.quadratic_beta_schedule(
                beta_start, beta_end, timesteps
            )
        elif schedule_type == "cosine":
            self.schedule = self.cos_beta_schedule(beta_start, beta_end, timesteps)
        else:
            raise ValueError(
                f"Schedule type {schedule_type} not supported. Please choose from linear, sigmoid, quadratic, or cosine."
            )
