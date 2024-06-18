import numpy as np
import matplotlib.pyplot as plt

from defdap.quat import Quat
import defdap.ebsd as ebsd
from defdap.plotting import MapPlot

from beta_reconstruction.reconstruction import (
    do_reconstruction, load_map, assign_beta_variants, 
    construct_variant_map, construct_beta_quat_array, 
    create_beta_ebsd_map)

def plot_beta(variant_map, beta_quat_array, direction, **kwargs):
    beta_IPF_colours = Quat.calcIPFcolours(
        beta_quat_array[variant_map >= 0],
        direction,
        "cubic"
    ).T

    map_colours = np.zeros(ebsd_map.shape + (3,))
    map_colours[variant_map >= 0] = beta_IPF_colours
    # recolour the -1 and -2 variants
    # -1 grains not succesfully reconstructed (white)
    # -2 clusters too small to be a grain and other phases (black)
    map_colours[variant_map == -1] = np.array([1, 1, 1])
    map_colours[variant_map == -2] = np.array([0, 0, 0])

    return MapPlot.create(ebsd_map, map_colours, **kwargs)

ebsd_file_path = "A01_XY_100_nm.ctf"

boundary_tolerance = 3
min_grain_size = 3

ebsd_map = load_map(
    ebsd_file_path,
    boundary_tolerance=boundary_tolerance,
    min_grain_size=min_grain_size
)
alpha_phase_id = 0
beta_phase_id = 1

print('alpha phase name: ', ebsd_map.phases[alpha_phase_id].name)
print('beta phase name: ', ebsd_map.phases[beta_phase_id].name)


ipf_dir = np.array([0, 0, 1])

ebsd_map.plotIPFMap(ipf_dir, phases=[alpha_phase_id])
plt.show()

ebsd_map.plotIPFMap(ipf_dir, phases=[beta_phase_id])
plt.show()

do_reconstruction(
    ebsd_map,
    burg_tol=1.,
    ori_tol=4.,
    alpha_phase_id=alpha_phase_id,
    beta_phase_id=beta_phase_id
)
assign_beta_variants(ebsd_map, "modal", alpha_phase_id=alpha_phase_id)

variant_map = construct_variant_map(ebsd_map, alpha_phase_id=alpha_phase_id)
beta_quat_array = construct_beta_quat_array(ebsd_map, variant_map=variant_map)

plot_beta(variant_map, beta_quat_array, np.array([0,0,1]))
plt.show()

ebsd_map.locateGrainID()

#grain_id = 8
#grain = ebsd_map[grain_id]
#grain.beta_oris

#np.array(grain.beta_deviations) *180 /np.pi
#grain.possible_beta_oris
#grain.variant_count

def click_print_beta_info(event, plot):
    if event.inaxes is not plot.ax:
        return

    # grain id of selected grain
    currGrainId = int(plot.callingMap.grains[int(event.ydata), int(event.xdata)] - 1)
    if currGrainId < 0:
        return

    # update the grain highlights layer in the plot
    plot.addGrainHighlights([currGrainId], alpha=plot.callingMap.highlightAlpha)

    # Print beta info
    grain = plot.callingMap[currGrainId]
    print("Grain ID: {}".format(currGrainId))
    print("Phase name:", grain.phase.name)
    print("Possible beta oris:", grain.possible_beta_oris)
    print("Beta deviations", np.rad2deg(grain.beta_deviations))
    print("Variant count", grain.variant_count)
    print("Assigned variant", grain.assigned_variant)
    print()

# Assign the plotting function to use with `locateGrainID`
ebsd_map.plotDefault = plot_beta

plot = ebsd_map.locateGrainID(
    variant_map=variant_map,
    beta_quat_array=beta_quat_array,
    direction=np.array([0, 0, 1]),
    clickEvent=click_print_beta_info
)
plt.show()
import pdb;pdb.set_trace()

ebsd_map_recon = create_beta_ebsd_map(
    ebsd_map,
    mode='alone',
    beta_quat_array=beta_quat_array,
    variant_map=variant_map,
    alpha_phase_id=alpha_phase_id,
    beta_phase_id=beta_phase_id
)
ebsd_map_recon.save('recon_map')

import pdb;pdb.set_trace()
