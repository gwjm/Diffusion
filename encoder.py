import torch.nn as nn
import torch
import numpy as np

"""
The goal of this is to understand how a diffusion encoder works

Assumptions: 
For Simplicity, I'm going to only use one beta value, and assume a standard normal distribution for each timestep with mean of 0 and sd of 1. 
"""


class SimpleDiffusionEncoder(nn.Module):
    def __init__(self, Beta: torch.float) -> None:
        super().__init__()
        self.beta = Beta

    def forward(self, x: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        Params:
        x: torch.Tensor: Input image
        timestep: int: number of timesteps to step forward in the noising process

        Adds noise sampled from the standard normal distribution to the image recursively
        """
        """ TODO:
        Change this function to add random noise to only the image color channels, not the whole tensor
        """
        if timestep == 0:
            return x

        x[0] *= torch.sqrt(torch.full((1, 1), 1.0 - self.beta))
        noise = torch.sqrt(torch.full((1, 1), self.beta)) * (
            torch.normal(0, 1, size=([x.shape[1], x.shape[2]])) / 256.0
        )
        x[0] += noise

        timestep -= 1
        if timestep == 0:
            return x
        else:
            return self.forward(x, timestep)
