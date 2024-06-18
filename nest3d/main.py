import numpy as np
import matplotlib.pyplot as plt

import defdap.hrdic as hrdic
import defdap.ebsd as ebsd
from defdap.quat import Quat
from beta_reconstruction.reconstruction import (
    do_reconstruction, load_map, assign_beta_variants,
    construct_variant_map, construct_beta_quat_array,
    create_beta_ebsd_map)

from beta_visualization import plot_beta, click_print_beta_info

if __name__ == "__main__":

    ebsdFilePath = "data/preprocessing/ctf_files/A01_XY_100_nm"
    ebsdMap = ebsd.Map(ebsdFilePath, dataType="OxfordText")
    ebsdMap.buildQuatArray()


    ebsdMap.plotEulerMap(plotScaleBar=True)
    plt.show()

    ebsdMap.plotIPFMap([1,0,0], plotScaleBar=True)
    plt.show()

    ebsdMap.plotKamMap(vmin=0, vmax=1)
    plt.show()

    ebsdMap.findBoundaries(boundDef=3)
    ebsdMap.findGrains(minGrainSize=3)

    for phase in ebsdMap.phases:
        print(phase.name)
        phase.printSlipSystems()

    ebsdMap.calcAverageGrainSchmidFactors(loadVector=np.array([1,0,0]))

    ebsdMap.plotAverageGrainSchmidFactorsMap()
    plt.show()

    ebsdMap.buildNeighbourNetwork()
    plt.show()
    
    import pdb;pdb.set_trace()
