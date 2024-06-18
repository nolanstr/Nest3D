import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob

import defdap.hrdic as hrdic
import defdap.ebsd as ebsd
from defdap.quat import Quat
from beta_reconstruction.reconstruction import (
    do_reconstruction,
    load_map,
    assign_beta_variants,
    construct_variant_map,
    construct_beta_quat_array,
    create_beta_ebsd_map,
    multiprocess_do_reconstruction,
)

from util.beta_visualization import (
    plot_beta,
    plot_alpha,
    plot_alpha_variant,
    plot_beta_IPF,
    click_print_beta_info,
    click_print_alpha_info,
)
from util.union_find import merge_sets
from util.identify_beta_grains import identify_beta_grains, label_alpha_variants
from util.crystallography import HCPVariants


def check_outputs(outputs):
    if not os.path.isdir(outputs):
        os.system(f"mkdir {outputs}")
        os.system(f"mkdir {outputs}/pngs")
        os.system(f"mkdir {outputs}/alpha_pngs")


def main(
    ebsdFilePath,
    outputs,
    boundDef=3,
    minGrainSize=10,
    alpha_phase_id=0,
    ori_tol=1.0,
    burg_tol=np.inf,
    beta_phase_id=1,
    savefigs=True,
    showfigs=False,
    load_pickle=False,
):
    if "XY" in ebsdFilePath:
        loadVector = np.array([1, 0, 0])
    elif "XZ" in ebsdFilePath:
        loadVector = np.array([0, 1, 0])
    elif "YZ" in ebsdFilePath:
        loadVector = np.array([0, 0, 1])
    else:
        raise ValueError("Expected XY, XZ, or YZ in EBSD filename!")
    check_outputs(outputs)

    ebsdMap = ebsd.Map(ebsdFilePath, dataType="OxfordText")
    ebsdMap.buildQuatArray()
    hcp = HCPVariants()
    hcpQuats = [Quat.fromEulerAngles(*eulerAngles) for eulerAngles in hcp.euler_angles]

    ebsdMap.findBoundaries(boundDef=boundDef)
    ebsdMap.findGrains(minGrainSize=minGrainSize)
    ebsdMap.calcAverageGrainSchmidFactors(loadVector=loadVector)
    ebsdMap.buildNeighbourNetwork()

    do_reconstruction(
        ebsdMap,
        burg_tol=burg_tol,
        ori_tol=ori_tol,
        alpha_phase_id=alpha_phase_id,
        beta_phase_id=beta_phase_id,
    )

    if load_pickle:
        print(f"Loading pickle from {outputs}...")
        f = open(f"{outputs}/stored_ebsdMap.pkl", "rb")
        ebsdMap = pickle.load(f)
        f.close()
    else:
        print(f"Storing pickle from {outputs}...")
        f = open(f"{outputs}/stored_ebsdMap.pkl", "wb")
        pickle.dump(ebsdMap, f)
        f.close()

    assign_beta_variants(ebsdMap, "modal", alpha_phase_id=alpha_phase_id)
    variant_map = construct_variant_map(ebsdMap, alpha_phase_id=alpha_phase_id)
    beta_quat_array = construct_beta_quat_array(ebsdMap, variant_map=variant_map)

    plot_beta_IPF(ebsdMap, variant_map, beta_quat_array, np.array([0, 0, 1]))
    if savefigs:
        plt.savefig(f"{outputs}/pngs/beta_IPF_plot", dpi=300)
    if showfigs:
        plt.show()
    plt.clf()

    ebsdMap.locateGrainID()

    # Assign the plotting function to use with `locateGrainID`
    ebsdMap.plotDefault = plot_beta_IPF
    plot = ebsdMap.locateGrainID(
        ebsd_map=ebsdMap,
        variant_map=variant_map,
        beta_quat_array=beta_quat_array,
        direction=np.array([0, 0, 1]),
        clickEvent=click_print_beta_info,
    )
    if savefigs:
        plt.savefig(f"{outputs}/pngs/beta_click_plot", dpi=300)
    if showfigs:
        plt.show()
    plt.clf()

    identify_beta_grains(ebsdMap)
    label_alpha_variants(ebsdMap, hcpQuats)
    plot = ebsdMap.locateGrainID(
        ebsd_map=ebsdMap,
        variant_map=variant_map,
        beta_quat_array=beta_quat_array,
        direction=np.array([0, 0, 1]),
        clickEvent=click_print_alpha_info,
    )
    if savefigs:
        plt.savefig(f"{outputs}/pngs/alpha_click_plot", dpi=300)
    if showfigs:
        plt.show()
    plt.clf()

    plot_beta(ebsdMap)
    if savefigs:
        plt.savefig(f"{outputs}/pngs/beta_grains_plot", dpi=300)
    if showfigs:
        plt.show()
    plt.clf()

    plot_alpha(ebsdMap)
    if savefigs:
        plt.savefig(f"{outputs}/pngs/alpha_grains_plot", dpi=300)
    if showfigs:
        plt.show()
    plt.clf()

    for i in range(12):
        plot_alpha_variant(ebsdMap, i)
        if savefigs:
            plt.savefig(f"{outputs}/alpha_pngs/alpha_{i}", dpi=300)
        if showfigs:
            plt.show()
        plt.clf()
    else:
        print(f"Storing pickle from {outputs}...")
        f = open(f"{outputs}/final_ebsdMap.pkl", "wb")
        pickle.dump(ebsdMap, f)
        f.close()


if __name__ == "__main__":
    ebsdFilePaths = glob.glob("./preprocess_ebsd/ctf_files/*")
    for ebsdFilePath in ebsdFilePaths:
        ebsdFilePath = ebsdFilePath.split(".c")[0]
        outputs = f"outputs/{ebsdFilePath.split('/')[-1]}"
        main(ebsdFilePath, outputs)
