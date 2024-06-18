import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_3d_volume(volume):
    """
    Plot a 3D volume using Matplotlib.

    volume: 3D numpy array or tensor
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.squeeze().cpu().numpy()  # Convert to numpy array if tensor and remove batch/channel dims

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the indices of the non-zero elements
    z, y, x = np.nonzero(volume)

    # Get the values (identifiers) of the non-zero elements
    values = volume[z, y, x]

    # Scatter plot
    sc = ax.scatter(x, y, z, c=values, cmap='viridis', marker='o')

    # Color bar to show the mapping of values to colors
    plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()



def create_test_volume(shape, num_ellipsoids=3):
    volume = torch.zeros(shape)
    identifiers = list(range(1, num_ellipsoids + 1))

    for identifier in identifiers:
        centroid = (
            np.random.uniform(0, shape[0] - 1),
            np.random.uniform(0, shape[1] - 1),
            np.random.uniform(0, shape[2] - 1),
        )
        radii = (np.random.uniform(3, 8), np.random.uniform(3, 8), np.random.uniform(3, 8))
        orientation_angle = np.random.uniform(0, 2 * np.pi)
        orientation = torch.tensor(
            [
                [np.cos(orientation_angle), -np.sin(orientation_angle), 0],
                [np.sin(orientation_angle), np.cos(orientation_angle), 0],
                [0, 0, 1],
            ],
            dtype=torch.double,
        )

        mask = generate_ellipsoidal_mask(shape, centroid, radii, orientation)
        volume[mask.bool()] = identifier

    return volume

def generate_ellipsoidal_mask(shape, centroid, radii, orientation):
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
        torch.linspace(-1, 1, shape[2], dtype=torch.double) 
    )
    grid = torch.stack([x, y, z], dim=-1)
    grid = grid @ orientation
    
    # Translate the grid to the centroid
    grid = grid - torch.tensor(centroid).float()
    
    # Normalize grid coordinates by the radii
    normalized_grid = grid / torch.tensor(radii).float()
    
    ellipsoid = (normalized_grid[..., 0]**2 + normalized_grid[..., 1]**2 + normalized_grid[..., 2]**2) <= 1
    return ellipsoid.float()

if __name__ == "__main__":
    test_volume = create_test_volume((32, 32, 32))
    import pdb;pdb.set_trace()
    plot_3d_volume(test_volume)
