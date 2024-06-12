import numpy as np
import time
import sys
from scipy.spatial.transform import Rotation as R

def misorientation_neighbors(M, neighbors, sel=None, out="deg", **kwargs):
    """
    Calculates the misorientation angle of every data point with respective
    orientation matrix provided in 'M' with respect to an arbitrary number
    of neighbors, whose indices are provided in the 'neighbors' argument.

    Parameters
    ----------
    M : numpy ndarray shape(N, 3, 3)
        List of rotation matrices describing the rotation from the sample
        coordinate frame to the crystal coordinate frame
    neighbors : numpy ndarray shape(N, K) - K being the number of neighbors
        Indices of the neighboring pixels
    sel : bool numpy 1D array (optional)
        Boolean array indicating data points calculations should be
        performed
        Default: None
    out : str (optional)
        Unit of the output. Possible values are:
        'deg': angle(s) in degrees
        'rad': angle(s) in radians
        Default: 'deg'
    **kwargs :
        verbose : bool (optional)
            If True, prints computation time
            Default: True

    Returns
    -------
    misang : numpy ndarray shape(N, K) - K being the number of neighbors
        KAM : numpy ndarray shape(N) with KAM values
    """
    N = M.shape[0]
    nneighbors = neighbors.shape[1]

    #C = list_cubic_symmetry_operators()
    C = get_hcp_symmetry_rotations()
    # 2D array to store trace values initialized as -2 (trace values are
    # always in the [-1, 3] interval)
    tr = np.full((N, nneighbors), -2.0, dtype=float)
    # 2D array to store the misorientation angles in degrees
    misang = np.full((N, nneighbors), -1.0, dtype=float)

    if not isinstance(sel, np.ndarray):
        sel = np.full(N, True, dtype=bool)

    verbose = kwargs.pop("verbose", True)
    if verbose:
        t0 = time.time()
        sys.stdout.write(
            "Calculating misorientations for {} points for {} neighbors".format(
                np.count_nonzero(sel), nneighbors
            )
        )
        sys.stdout.write(" [")
        sys.stdout.flush()

    for k in range(nneighbors):
        # valid points, i.e., those part of the selection and with valid neighrbor index (> 0)
        ok = (neighbors[:, k] >= 0) & sel & sel[neighbors[:, k]]
        # Rotation from M[ok] to M[neighbors[ok, k]]
        # Equivalent to np.matmul(M[neighbors[ok,k]], M[ok].transpose([0,2,1]))
        T = np.einsum("ijk,imk->ijm", M[neighbors[ok, k]], M[ok])

        for m in range(len(C)):
            # Smart way to calculate the trace using einsum.
            # Equivalent to np.matmul(C[m], T).trace(axis1=1, axis2=2)
            a, b = C[m].nonzero()
            ttr = np.einsum("j,ij->i", C[m, a, b], T[:, a, b])
            tr[ok, k] = np.max(np.vstack([tr[ok, k], ttr]), axis=0)

        if verbose:
            if k > 0 and k < nneighbors:
                sys.stdout.write(", ")
            sys.stdout.write("{}".format(k + 1))
            sys.stdout.flush()

    del T, ttr

    if verbose:
        sys.stdout.write("] in {:.2f} s\n".format(time.time() - t0))
        sys.stdout.flush()

    # Take care of tr > 3. that might happend due to rounding errors
    tr[tr > 3.0] = 3.0

    # Filter out invalid trace values
    ok = tr >= -1.0
    misang[ok] = trace_to_angle(tr[ok], out)
    return misang

def trace_to_angle(tr, out="deg"):
    """
    Converts the trace of a orientation matrix to the misorientation angle
    """
    ang = np.arccos((tr - 1.0) / 2.0)
    if out == "deg":
        ang = np.degrees(ang)
    return ang

def get_hcp_symmetry_rotations():
    rotations = []

    # 6-fold rotations about the c-axis
    for angle in range(0, 360, 60):
        rotation = R.from_euler('z', angle, degrees=True)
        rotations.append(rotation)

    # 3-fold rotations about the c-axis
    for angle in range(0, 360, 120):
        rotation = R.from_euler('z', angle, degrees=True)
        rotations.append(rotation)

    # 2-fold rotations about axes perpendicular to the c-axis
    rotations.append(R.from_euler('x', 180, degrees=True))
    rotations.append(R.from_euler('y', 180, degrees=True))
    rotations.append(R.from_euler('z', 180, degrees=True))
    rotations = np.array([rot.as_matrix() for rot in rotations])
    return rotations
