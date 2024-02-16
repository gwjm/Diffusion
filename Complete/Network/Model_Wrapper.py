from utils.encoder import Q_Sampler
import torch
import torch.nn as nn

# Sampler Abstraction
# Forward, backward, and sampling in a single model.
# Model is taking the noise and predicting the specific noise.
# If we can learn the mean we can reparameterize to find the exact the noid


class Model_Wrapper(nn.Module):
    def __init__(
        self,
        beta_schedule: torch.Tensor,
        max_timesteps: torch.Tensor,
        decoder: nn.Module,
    ) -> None:

        # Define the Forward Process object
        self.Q = Q_Sampler(beta_schedule, max_timesteps)

        # Define the U-Net
        self.decoder = decoder

    def q_sample_image(self, image: torch.Tensor):
        return self.Q(image)

    def forward(self, x):
        pass

    def sample_new_image(self):
        pass
