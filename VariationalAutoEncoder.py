import torch.nn as nn
from torch.nn import functional as F
from torch import tensor as Tensor
import torch


class VariationalAutoEncoder(nn.Module):
    def __init__(
        self, in_channels: int, latent_dim: int, hidden_dims: list = None, **kwargs
    ) -> None:
        super(VariationalAutoEncoder, self).__init__()
        modules = []
        if hidden_dims == None:
            hidden_dims = [32, 64, 128, 256, 512]

        for dim in hidden_dims:
            modules.append(
                nn.sequential(
                    nn.conv2d(
                        in_channels,
                        out_channels=dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        modules = []

        hidden_dims.reverse()
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, input: Tensor) -> list[Tensor]:
        mu, logvar = self.endcode(input)

        z = self.reparameterize(mu, logvar)

        return [self.decode(z), input, mu, logvar]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def loss(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": -kld_loss}

    def generate(self, x) -> Tensor:
        return self.forward(x)[0]

    def encode(self, input: Tensor) -> list[Tensor]:
        result = torch.flatten(self.encoder(input), start_dim=1)
        self.fc_mu = self.fc_mu(result)
        self.fc_var = self.fc_var(result)

        return [self.fc_mu, self.fc_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
