import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visualization import plot_3d_volume
print("Need to randomly sample a beta orientation as well!!!!!!1")
class EllipsoidNet(nn.Module):
    def __init__(self):
        super(EllipsoidNet, self).__init__()
        self.variant_ids = np.arange(1, 13)
        # Define convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Define fully connected layers for regression
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * (input_volume_size // 8) ** 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(
                128, 12 + 3 + 3
            ),  # 12 for identifier, 3 for a, b, c, 3 for xc, yc, zc
        )

    def forward(self, x):
        # Forward pass through convolutional layers
        out = self.conv_layers(x)
        # Flatten the output of convolutional layers
        out = out.view(out.size(0), -1)
        # Forward pass through fully connected layers
        out = self.fc_layers(out)
        return out

    def generate_ellipsoidal_mask(self, shape, centroid, radii, orientation):
        """
        Generates a mask for an ellipsoid with a given centroid, radii, and orientation.

        shape: tuple of (depth, height, width)
        centroid: tuple of (z, y, x)
        radii: tuple of (radius_z, radius_y, radius_x)
        orientation: assumed orientation basis (e.g., as a rotation matrix)

        Returns: A binary mask with ellipsoidal regions
        """
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, shape[0], dtype=torch.double),
            torch.linspace(-1, 1, shape[1], dtype=torch.double),
            torch.linspace(-1, 1, shape[2], dtype=torch.double),
        )
        grid = torch.stack([x, y, z], dim=-1)
        grid = grid @ orientation

        # Translate the grid to the centroid
        grid = grid - torch.tensor(centroid).float()

        # Normalize grid coordinates by the radii
        normalized_grid = grid / torch.tensor(radii).float()

        ellipsoid = (
            normalized_grid[..., 0] ** 2
            + normalized_grid[..., 1] ** 2
            + normalized_grid[..., 2] ** 2
        ) <= 1
        return ellipsoid.float()

    def fill_ellipsoid(self, x, output):
        # Separate the output into identifier, a, b, c, xc, yc, zc
        identifier = output[:, :12]
        a_b_c = output[:, 12:15]  # Contains a, b, c
        xc_yc_zc = output[:, 15:]  # Contains xc, yc, zc
        print(f"a, b, c params: {a_b_c}")
        print(f"xc, yc, and zx params: {xc_yc_zc}")
        shape =  x.shape[2:]
        orientation = torch.eye(3, dtype=torch.double)
        ellipsoid = self.generate_ellipsoidal_mask(shape, xc_yc_zc, 1.0,
                                                   orientation)
        output_prob = F.softmax(identifier, dim=1)

        output_prob = output_prob / output_prob.sum(dim=1, keepdim=True)
        output_prob = output_prob.detach().numpy().flatten()
        variant_id = np.random.choice(self.variant_ids, p=output_prob)
        x[0,0][ellipsoid.bool()] = variant_id
        return x


# Example usage
# Assuming input_volume is your input volume tensor of shape (batch_size, channels, depth, height, width)
input_volume = torch.zeros(1, 1, 32, 32, 32)  # Example input volume
in_channels = 1  # Assuming single channel input volume
input_volume_size = 32  # Assuming input volume size of 32x32x32

# Instantiate the model
model = EllipsoidNet()


n_passes = 5
# Forward pass
for _ in range(n_passes):
    output = model(input_volume)
    input_volume = model.fill_ellipsoid(input_volume, output)
plot_3d_volume(input_volume)

import pdb

pdb.set_trace()
