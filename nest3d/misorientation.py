import numpy as np

def misorientation_angle(O1, O2):
    M = O1 @ np.linalg.inv(O2)
    theta = np.arccos((M.trace()-1)/2)
    return theta

def check_neighbors(grain, ebsdMap, tol=1.0):
    
    neighbors = ebsdMap.neighbourNetwork[grain]
    group = [grain.grainID]
    for n in neighbors:
        misori = grain.refOri.misOri(n.refOri, grain.crystalSym)
        if misori <= tol:
            group.append(n.grainID)

    return group
