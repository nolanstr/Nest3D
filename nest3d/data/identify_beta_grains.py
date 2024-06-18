import numpy as np
from scipy.spatial.transform import Rotation 
from defdap.quat import Quat

from union_find import merge_sets
from beta_util import Beta

def identify_beta_grains(ebsdMap,
                           TOL=4.5, 
                           sizeTOL=1000, 
                           nLathsTOL=10):
    premerge_beta_groups = []
    for i, grain in enumerate(ebsdMap):
        if hasattr(grain, "beta_deviations"):
            if len(grain.beta_deviations) > 0:
                neighbours = ebsdMap.neighbourNetwork[grain]
                deviations = np.array(grain.beta_deviations)*180.0/np.pi
                phases = np.array([n.phaseID for n in neighbours])
                neighbourIDs = np.array([n.grainID for n in neighbours])
                checkNeighbours = neighbourIDs[phases==grain.phaseID][deviations<=TOL]
                if checkNeighbours.sum() > 0:
                    premerge_beta_groups.append(set([grain.grainID] + checkNeighbours.tolist()))
    
    premerge_beta_groups = merge_sets(premerge_beta_groups)
    laths_per_grain = np.array([len(group) for group in premerge_beta_groups])
    sizes = np.array([sum([len(ebsdMap[gID].coordList) for gID in list(group)]) \
                                    for group in premerge_beta_groups])
    beta_groups = [] 
    sizeTOL = 1000
    nLathsTOL = 10
    for group, group_size, n_laths in zip(premerge_beta_groups, sizes, laths_per_grain):
        if (group_size >= sizeTOL) and (n_laths >= nLathsTOL):
            beta_groups.append(group)
    beta_grain_ids = np.arange(len(beta_groups)).astype(int)
    beta_grains = [] 
    print(f"Organizing parent beta objects...")
    for beta_grain_id in beta_grain_ids:
        print(f"Beta {beta_grain_id} running...")
        beta_grains.append(Beta(ebsdMap, beta_groups[beta_grain_id],
                                beta_grain_id))
        print(f"Beta {beta_grain_id} completed!")
    
    ebsdMap.beta_grains = beta_grains

def label_alpha_variants(ebsdMap, rotQuats):
    
    rotRMatrix = np.array([quat.rotMatrix() for quat in rotQuats])

    for beta in ebsdMap.beta_grains:
        betaR = beta.refOri.rotMatrix()
        alphaRotMatrix = np.einsum("mi,kij,nj->kmn", betaR, rotRMatrix, betaR)
        alphaQuats = [Quat.fromEulerAngles(*Rotation.from_matrix(mat).as_euler("ZXZ"))\
                                                        for mat in alphaRotMatrix]
        alphaMisOris = np.array([beta.compute_beta_oris_misOri(quat).min(axis=0) \
                            for quat in alphaQuats])
        beta.alpha_variant_ids = np.argmin(alphaMisOris, axis=0)

    assign_alpha_variant_ids(ebsdMap)

def assign_alpha_variant_ids(ebsdMap):

    for beta in ebsdMap.beta_grains:
        for i, grain in enumerate(beta.alpha_grains):
            grain.alpha_id = beta.alpha_variant_ids[i]


