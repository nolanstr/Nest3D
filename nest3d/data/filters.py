import pyebsd

from union_find import merge_sets
from custom_misorientation_neighbors import *

def filter_beta_grains(scan, hcp_misorientations, TOL=1.0):
    dims = (scan.nrows, scan.ncols)

    neighbors = scan.get_neighbors(distance=1)
    non_alpha_idxs = np.arange(scan.phase.shape[0])[scan.phase!=2]
    mask = np.isin(neighbors, non_alpha_idxs)
    indices = np.where(mask)
    neighbors[indices] = -1

    Rs = scan.R
    phases = scan.phase
    neighbor_misoris = misorientation_neighbors(scan.R,
                                                       neighbors)
    neighbor_misoris[neighbor_misoris==-1] = np.nan

    diff = neighbor_misoris[:,:,None] - hcp_misorientations
    checks = np.any(diff<TOL, axis=2)
    beta_grains = filter_checks_and_neighbors(checks, neighbors)
    image = update_image_by_groups(beta_grains, dims, phases)
    X, Y = np.meshgrid(scan.x, scan.y)
    return beta_grains, image, X, Y


def filter_alpha_grains(scan, TOL=1.0):
    dims = (scan.nrows, scan.ncols)

    neighbors = scan.get_neighbors(distance=1)
    non_alpha_idxs = np.arange(scan.phase.shape[0])[scan.phase!=2]
    mask = np.isin(neighbors, non_alpha_idxs)
    indices = np.where(mask)
    neighbors[indices] = -1

    Rs = scan.R
    phases = scan.phase
    neighbor_misoris = misorientation_neighbors(scan.R,
                                                       neighbors)
    neighbor_misoris[neighbor_misoris==-1] = np.nan

    diff = neighbor_misoris[:,:,None]
    checks = np.any(diff<TOL, axis=2)
    alpha_grains = filter_checks_and_neighbors(checks, neighbors)
    image = update_image_by_groups(alpha_grains, dims, phases)
    X, Y = np.meshgrid(np.arange(dims[0]),
                       np.flip(np.arange(dims[1])))
    return alpha_grains, image, X, Y


def update_image_by_groups(groups, dims, phases):
    image = np.zeros((dims[0], dims[1]))
    for group_i, group in enumerate(groups):
        i, j = get_idxs(np.array(list(group)), dims)
        image[i,j] = group_i + 1
    for i, phase in enumerate(phases):
        if phase != 2:
            i, j = get_idxs(i, dims)
            image[i, j] = np.nan
    image[image==0] = np.nan
    return image

def get_idxs(ids, dims):
    i = ids // dims[0]
    j = ids - dims[0]*i
    return i, j

def filter_checks_and_neighbors(checks, neighbors):

    idxs = np.append(0, np.cumsum(np.sum(checks, axis=1)))
    flattened_values = neighbors[checks]
    sets = [set(flattened_values[idxs[i]:idxs[i+1]].tolist())\
                        for i in range(idxs.shape[0]-1)]
    beta_grain_sets = merge_sets(sets)
    return beta_grain_sets

def neighbor_filter(arr):
    filtered_arr = arr.copy()
    for sublist in range(len(arr)):
        for i in range(len(arr[sublist])):
            # Check if the current element is different from its neighbors
            if (i > 0 and arr[sublist][i] != arr[sublist][i - 1]) or \
               (i < len(arr[sublist]) - 1 and arr[sublist][i] != arr[sublist][i + 1]):
                # Replace the current element with the value of its neighbors
                if i > 0:
                    filtered_arr[sublist][i] = arr[sublist][i - 1]
                if i < len(arr[sublist]) - 1:
                    filtered_arr[sublist][i] = arr[sublist][i + 1]
    return filtered_arr
