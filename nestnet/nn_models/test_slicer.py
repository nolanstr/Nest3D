import torch

from generator import EllipsoidNet
from slicer import Slicer
from visualization import plot_3d_volume

if __name__ == "__main__":
    # Example usage
    n = 32  # Assuming input volume size of 32x32x32
    # Assuming input_volume is your input volume tensor of shape 
    # (batch_size, channels, depth, height, width)
    input_volume = torch.arange(n**3).reshape((n,n,n))  # Example input volume
    slicer = Slicer([10,20])
    slices = slicer(input_volume, sample=10)

    for key1 in slices.keys():
        for key2 in slices[key1].keys():
            print(f"(samples, dim, dim) = {slices[key1][key2].shape}")
