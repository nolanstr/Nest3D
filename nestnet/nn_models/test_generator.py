import torch

from generator import EllipsoidNet
from visualization import plot_3d_volume

if __name__ == "__main__":
    # Example usage
    input_size = 32  # Assuming input volume size of 32x32x32
    # Assuming input_volume is your input volume tensor of shape 
    # (batch_size, channels, depth, height, width)
    input_volume = torch.zeros(1, 1, input_size, input_size, input_size)  # Example input volume
    in_channels = 1  # Assuming single channel input volume
    stop = False
    # Instantiate the model
    model = EllipsoidNet(input_size, in_channels)


    # Forward pass
    input_volume = model(input_volume)
    plot_3d_volume(input_volume)

    import pdb

    pdb.set_trace()
