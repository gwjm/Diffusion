from utils.encoder import SimpleDiffusionEncoder
from torch import random
import matplotlib.pyplot as plt
from torchvision.io import read_image


def main():
    random.manual_seed(42)
    image = read_image(
        "/Users/Gregor/Code/Personal Projects/Python/Diffusion/image.png"
    )
    f, axarr = plt.subplots(1, 3)

    axarr[0].imshow(image.permute((1, 2, 0)))
    noise_image = image.float()
    encoder = SimpleDiffusionEncoder(0.25)
    noise_image /= 256.0
    image = encoder.forward(noise_image, 20)
    noise_image *= 256.0
    noise_image = noise_image.byte()
    axarr[1].imshow(noise_image.permute(1, 2, 0))
    axarr[2].imshow((noise_image - image).permute(1, 2, 0))
    plt.show()


main()
