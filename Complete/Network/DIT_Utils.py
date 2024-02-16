import torch.nn as nn
from torch import Tensor


class Patchify(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super(Patchify, self).__init__()
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_channels, height, width = x.shape
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), "Image size must be divisible by the patch size"
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch_size, -1, n_channels, self.patch_size, self.patch_size
        )
        return patches
