import numpy as np
import matplotlib.pyplot as plt

import defdap.hrdic as hrdic
import defdap.ebsd as ebsd
from defdap.quat import Quat
from beta_reconstruction.reconstruction import (
    do_reconstruction, load_map, assign_beta_variants,
    construct_variant_map, construct_beta_quat_array,
    create_beta_ebsd_map)

import pickle
from beta_visualization import plot_beta, click_print_beta_info

if __name__ == "__main__":
    alpha_phase_id = 0
    beta_phase_id = 1
    f = open("stored_ebsd_map.pkl", "rb")
    ebsdMap = pickle.load(f)
    f.close()
    variant_map = construct_variant_map(ebsdMap, alpha_phase_id=alpha_phase_id)
    beta_quat_array = construct_beta_quat_array(ebsdMap, variant_map=variant_map)
    plot_beta(ebsdMap, variant_map, beta_quat_array, np.array([0,0,1]))
    plt.show()
    import pdb;pdb.set_trace()
    ebsdMap.locateGrainID()

    betaGrainID = 2
    betaGrain = ebsdMap[betaGrainID]
    print(betaGrain.beta_oris)
    print(np.array(betaGrain.beta_deviations) *180 /np.pi)
    print(betaGrain.possible_beta_oris)
    print(betaGrain.variant_count)
    # Assign the plotting function to use with `locateGrainID`
    ebsdMap.plotDefault = plot_beta
    plot = ebsdMap.locateGrainID(
        ebsd_map=ebsdMap,
        variant_map=variant_map,
        beta_quat_array=beta_quat_array,
        direction=np.array([0, 0, 1]),
        clickEvent=click_print_beta_info
    )
    plt.show()

    ebsdMapRecon = create_beta_ebsd_map(
        ebsdMap,
        mode='alone',
        beta_quat_array=beta_quat_array,
        variant_map=variant_map,
        alpha_phase_id=alpha_phase_id,
        beta_phase_id=beta_phase_id
    )
    ebsdMapRecon.save("recon_map")
    import pdb;pdb.set_trace()
