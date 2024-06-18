import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon

maxFails = 50

def sample_squares(indexes, values, min_area, num_squares):
    
    squares = []
    fails = 0

    X = np.unique(indexes[:,0]).argsort()
    xLowerHalf = X[:X.shape[0]//2]
    xUpperHalf = X[X.shape[0]//2:]
    Y = np.unique(indexes[:,1]).argsort()
    
    while (len(squares)<num_squares) & (fails<maxFails):
        try:
            xMin = np.random.choice(xLowerHalf)
            xMax = np.random.choice(xUpperHalf)
            yInds = (xMin<=indexes[:,0]) & (indexes[:,0]<=xMax)
            yVals = indexes[yInds, 1]

            yLowerHalf = yVals[:yVals.shape[0]//2]
            yUpperHalf = yVals[yVals.shape[0]//2:]
            yMin = np.random.choice(yLowerHalf)
            yMax = np.random.choice(yUpperHalf)
            xInds = (yMin<=indexes[:,1]) & (indexes[:,1]<=yMax)
            Inds = xInds & yInds
            if Inds.sum() > 0:
                idxs = indexes[Inds]
                xVals = np.unique(idxs[:,0])
                yVals = np.unique(idxs[:,1])
                minVals = min([xVals.shape[0], xVals.shape[0]])
                xVals = xVals[:minVals]
                yVals = yVals[:minVals]
                Inds = np.isin(indexes[:,0], xVals) & \
                            np.isin(indexes[:,1], yVals)
                idxs = indexes[Inds,:]
                vals = values[Inds]
                if int(vals.shape[0]**0.5) == vals.shape[0]**0.5:
                    if vals.shape[0] >= min_area:
                        squares.append(vals)
        except:
            fails += 1
    
    if fails == maxFails:
        if len(squares) == 0:
            return False

    return squares

def sample_sized_squares(indexes, values, sizes, num_squares):
    
    squares = []
    fails = 0

    X = np.unique(indexes[:,0]).argsort()
    xLowerHalf = X[:X.shape[0]//2]
    #xUpperHalf = X[X.shape[0]//2:]
    Y = np.unique(indexes[:,1]).argsort()
    
    while (len(squares)<num_squares) & (fails<maxFails):
        try:
            idOffset = np.random.choice(sizes)

            xMin = np.random.choice(xLowerHalf)
            xMinId = np.argwhere(X == xMin)[0,0]
            xMaxId = xMinId +idOffset
            xMax = X[xMaxId]

            yInds = (xMin<=indexes[:,0]) & (indexes[:,0]<=xMax)
            yVals = indexes[yInds, 1]

            yLowerHalf = yVals[:yVals.shape[0]//2]
            yMin = np.random.choice(yLowerHalf)
            yMinId = np.argwhere(Y == yMin)[0,0]
            yMaxId = yMinId + idOffset
            yMax = Y[yMaxId]
            xInds = (yMin<=indexes[:,1]) & (indexes[:,1]<=yMax)
            Inds = xInds & yInds
            if Inds.sum() > 0:
                idxs = indexes[Inds]
                square = values[Inds]
                if int(square.shape[0]**0.5) == square.shape[0]**0.5:
                    squares.append(square)
        except:
            fails += 1
    
    if fails == maxFails:
        if len(squares) == 0:
            return False

    return squares

def extract_values_and_update_indexes(array, indexes):
    """
    Extract all values within the convex hull of the given set of indexes from a 450x450 array.
    Also update the set of indexes to include all points within the convex hull.

    Parameters:
    array (numpy.ndarray): A 450x450 array.
    indexes (list of tuples): A list of (row, col) index tuples.

    Returns:
    numpy.ndarray: Array of values within the convex hull of the given indexes.
    list of tuples: Updated set of indexes including all points within the convex hull.
    """
    # Convert list of tuples to a NumPy array
    points = np.array(indexes)
    
    # Calculate the convex hull of the points
    hull = ConvexHull(points)
    
    # Get the vertices of the convex hull
    hull_points = points[hull.vertices]
    
    # Create a mask for the convex hull
    rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], array.shape)
    
    # Extract values inside the convex hull
    values_within_hull = array[rr, cc]
    
    # Determine all points within the convex hull
    all_points_within_hull = list(zip(rr, cc))
    
    
    return values_within_hull, np.array(all_points_within_hull) 
