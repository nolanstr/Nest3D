import numpy as np
import matplotlib.pyplot as plt
from multiprocess import Pool
from tqdm import tqdm

from defdap.quat import Quat
import defdap.ebsd as ebsd
import defdap.hrdic as hrdic
from defdap.plotting import MapPlot

from filters import filter_beta_grains, filter_alpha_grains, neighbor_filter
from mergers import merge_small_groups
from util import compute_ND

if __name__ == "__main__":
    
    hcp_misorientations = np.load("hcp_misorientations.npy")
    hcp_misorientations = np.unique(hcp_misorientations).reshape((1,1,-1))
    boundary_tolerance=3
    min_grain_size=3
    scan = ebsd.Map("A01_XY_100_nm", 
                    dataType="OxfordText")
    import pdb;pdb.set_trace()
    crystal_normal = np.array([0,0,1])
    plane_normal = np.array([0,0,1])

    ND = compute_ND(scan, crystal_normal, plane_normal)[:,-1].reshape(scan.x.shape)

    import pdb;pdb.set_trace()
