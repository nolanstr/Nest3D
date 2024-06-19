
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EllipsoidNet(nn.Module):
    def __init__(self, input_size, in_channels=1, maxPasses=100):
        super(EllipsoidNet, self).__init__()
        self.alpha_orientations = torch.tensor(
                                    np.load("nn_models/hcp_variant_orientations.npy"),
                                    dtype=torch.double)
        self.variant_ids = np.arange(1, 13)
        self.set_beta_orientation()
        self.maxPasses = maxPasses
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
            nn.Linear(128 * (input_size // 8) ** 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(
                128, 12 + 3 + 3 + 1  # 12 for identifier, 3 for a, b, c, 3 for xc, yc, zc, 1 for EOS token
            ),
        )

    def forward(self, x):
        stop = False
        passes = 0
        while (not stop) and (passes<self.maxPasses):
            # Forward pass through convolutional layers
            out = self.conv_layers(x)
            # Flatten the output of convolutional layers
            out = out.view(out.size(0), -1)
            # Forward pass through fully connected layers
            out = self.fc_layers(out)
            # apply ellipsoid to volume x and check if it should stop.
            x, stop = self.fill_ellipsoid(x, out)
            passes += 1

        # Sets new beta orientation for next forward pass
        self.set_beta_orientation()

        return x

    def generate_ellipsoidal_mask(
        self, shape, centroid, radii, orientation,
    ):
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
        grid = grid @ self.beta_orientation @ orientation

        # Translate the grid to the centroid
        grid = grid - centroid.clone().detach()

        # Normalize grid coordinates by the radii
        normalized_grid = grid / radii.clone().detach()

        ellipsoid = (
            normalized_grid[..., 0] ** 2
            + normalized_grid[..., 1] ** 2
            + normalized_grid[..., 2] ** 2
        ) <= 1
        return ellipsoid.float()

    def fill_ellipsoid(self, x, output):
        # Separate the output into identifier, a, b, c, xc, yc, zc, EOS token
        identifier = output[:, :12] # Allow us to sample from variants.

        a_b_c = torch.abs(output[:, 12:15]) # Contains a, b, c
        # absolue makes them valid parameters!

        xc_yc_zc = output[:, 15:18]%1  # Contains xc, yc, zc
        # Modulo 1 shifts it into the cube!

        eos_token = output[:, 18]  # Contains EOS token

        output_prob = F.softmax(identifier, dim=1)
        output_prob = output_prob / output_prob.sum(dim=1, keepdim=True)
        output_prob = output_prob.detach().numpy().flatten()

        variant_id = np.random.choice(self.variant_ids, p=output_prob)
        alpha_orientation = self.alpha_orientations[variant_id - 1]
        ellipsoid = self.generate_ellipsoidal_mask(
            shape=(x.size(2), x.size(3), x.size(4)), 
            centroid=xc_yc_zc, 
            radii=a_b_c, 
            orientation=alpha_orientation
        )
        x[0, 0][ellipsoid.bool()] = variant_id
        return x, torch.sigmoid(eos_token).item() > 0.5  # Indicate that the process continues

    def set_beta_orientation(self):
        """
        Generate a random 3x3 rotation matrix for the parent beta phase orientation.

        """
        # Generate random quaternions
        rand_quaternions = torch.randn(4)
        rand_quaternions = rand_quaternions / rand_quaternions.norm(keepdim=True)

        # Convert quaternions to rotation matrices
        w, x, y, z = (
            rand_quaternions[0],
            rand_quaternions[1],
            rand_quaternions[2],
            rand_quaternions[3],
        )

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        self.beta_orientation = torch.zeros(3, 3, dtype=torch.double)
        self.beta_orientation[0, 0] = 1 - 2 * (yy + zz)
        self.beta_orientation[0, 1] = 2 * (xy - wz)
        self.beta_orientation[0, 2] = 2 * (xz + wy)
        self.beta_orientation[1, 0] = 2 * (xy + wz)
        self.beta_orientation[1, 1] = 1 - 2 * (xx + zz)
        self.beta_orientation[1, 2] = 2 * (yz - wx)
        self.beta_orientation[2, 0] = 2 * (xz - wy)
        self.beta_orientation[2, 1] = 2 * (yz + wx)
        self.beta_orientation[2, 2] = 1 - 2 * (xx + yy)


