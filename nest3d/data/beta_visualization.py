import numpy as np
from defdap.plotting import MapPlot
from defdap.quat import Quat
import matplotlib.pyplot as plt

def plot_beta(ebsd_map, cm=plt.cm.tab20):

    map_colours = np.zeros(ebsd_map.shape + (3,))
    
    for i, beta in enumerate(ebsd_map.beta_grains):
        for grain in beta.alpha_grains:
            idxs = np.array(grain.coordList)
            map_colours[idxs[:,0], idxs[:,1]] = list(cm(i))[:-1]

    return MapPlot.create(ebsd_map, map_colours)

def plot_alpha(ebsd_map, cm=plt.cm.tab20):
    map_colours = np.zeros(ebsd_map.shape + (3,))
    
    for beta in ebsd_map.beta_grains:
        for grain in beta.alpha_grains:
            idxs = np.array(grain.coordList)
            map_colours[idxs[:,0], idxs[:,1]] = list(cm(grain.alpha_id))[:-1]

    return MapPlot.create(ebsd_map, map_colours)

def plot_alpha_variant(ebsd_map, variant_id=0):
    map_colours = np.zeros(ebsd_map.shape + (3,))
    for beta in ebsd_map.beta_grains:
        for grain in beta.alpha_grains:
            if grain.alpha_id == variant_id:
                idxs = np.array(grain.coordList)
                map_colours[idxs[:,0], idxs[:,1]] = 255

    return MapPlot.create(ebsd_map, map_colours)


def plot_beta_IPF(ebsd_map, variant_map, beta_quat_array, direction, **kwargs):
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

def click_print_alpha_info(event, plot):
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
    print("Beta ID", grain.assigned_variant)
    print("Alpha Variant ID", grain.alpha_id)
    print()
