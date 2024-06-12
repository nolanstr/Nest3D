import numpy as np
from scipy.ndimage import label, generate_binary_structure
from scipy.spatial.distance import cdist

def merge_small_groups(image, threshold_size):
    # Label connected components
    nan_value = np.nanmin(image) - 1
    image[np.isnan(image)] = nan_value 
    
    structure = generate_binary_structure(2, 1)
    labeled_array, num_features = label(image, structure)

    #import pdb;pdb.set_trace()
    # Initialize dictionary to store group sizes
    group_sizes = {}
    for i in range(1, num_features + 1):
        group_sizes[i] = np.sum(labeled_array == i)

    # Iterate through each group
    for i in range(1, num_features + 1):
        if group_sizes[i] < threshold_size:
            # Find neighboring groups
            neighboring_groups = set()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    x, y = np.where(labeled_array == i)
                    x += dx
                    y += dy
                    valid_indices = (x >= 0) & (x < labeled_array.shape[0]) & (y >= 0) & (y < labeled_array.shape[1])
                    neighboring_groups.update(labeled_array[x[valid_indices], y[valid_indices]])

            # Find nearest larger group
            nearest_group = None
            min_distance = np.inf
            for group in neighboring_groups:
                if group_sizes[group] >= threshold_size:
                    distance = np.min(cdist(np.argwhere(labeled_array == i), np.argwhere(labeled_array == group)))
                    if distance < min_distance:
                        nearest_group = group
                        min_distance = distance

            # Merge small group with nearest larger group
            if nearest_group is not None:
                image[labeled_array == i] = nearest_group

    return image
