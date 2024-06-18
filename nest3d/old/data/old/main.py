import numpy as np
import matplotlib.pyplot as plt
import pyebsd
from multiprocess import Pool
from tqdm import tqdm

from filters import filter_beta_grains, filter_alpha_grains, neighbor_filter
from mergers import merge_small_groups
from util import compute_ND

if __name__ == "__main__":
    
    hcp_misorientations = np.load("hcp_misorientations.npy")
    hcp_misorientations = np.unique(hcp_misorientations).reshape((1,1,-1))
    scan = pyebsd.load_scandata("A01_XY_100_nm.ang")
    crystal_normal = np.array([0,0,1])
    plane_normal = np.array([0,0,1])

    ND = compute_ND(scan, crystal_normal, plane_normal)[:,-1].reshape(scan.x.shape)
    #normND = np.linalg.norm(ND, axis=1).reshape(scan.x.shape)
    scan.plot_property(ND, propname="Norm of Normal Direction")
    import pdb;pdb.set_trace()
    TOLS = [1,2,3,4,5,10,12]
    threshold_size = 50
    for TOL in TOLS:
        beta_grains, image, X, Y = filter_alpha_grains(scan, 
                                                      TOL=TOL)
        image = neighbor_filter(image)
        #image = merge_small_groups(image, threshold_size)
        fig, ax = plt.subplots()
        ax.contourf(X, Y, image)
        Z_nan = np.isnan(image)
        ax.contourf(X, Y, Z_nan, levels=[0.5, 1.5], colors='black')
        plt.savefig(f"./pngs/TOL_{TOL}", dpi=300)
        #plt.show()
        plt.clf()

    import pdb;pdb.set_trace()
