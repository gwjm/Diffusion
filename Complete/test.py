from utils.encoder import DiffusionEncoder
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import (
    Compose,
    ToTensor,
    Lambda,
    ToPILImage,
    CenterCrop,
    Resize,
)
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)


image_size = 128
transform = Compose(
    [
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),  # turn into torch Tensor of shape CHW, divide by 255
        Lambda(lambda t: (t * 2) - 1),
    ]
)

reverse_transform = Compose(
    [
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.0),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ]
)


def main():
    x = transform(image)
    encoder = DiffusionEncoder(beta_schedule=None, max_timesteps=1000)
    x_10 = encoder.forward(x, torch.Tensor([9]))
    x_100 = encoder.forward(x, torch.Tensor([99]))
    x_1000 = encoder.forward(x, torch.Tensor([999]))
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(reverse_transform(x))
    axarr[1].imshow(reverse_transform(x_10))
    axarr[2].imshow(reverse_transform(x_100))
    axarr[3].imshow(reverse_transform(x_1000))
    plt.show()


main()
