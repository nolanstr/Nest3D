import numpy as np
import torch

from discriminator import Discriminator

if __name__ == "__main__":

    n = 450

    discriminator = Discriminator(n)

    for _ in range(10):
        image = torch.round(torch.rand((n,n))*12)
        x = discriminator(image)
        print(f"Probability = {x}")
    import pdb;pdb.set_trace()
