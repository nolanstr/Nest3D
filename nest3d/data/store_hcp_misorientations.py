import numpy as np
import pyebsd

from crystallography import HCPVariants


if __name__ == "__main__":
    hcp = HCPVariants()
    misorientations = []

    for i in range(hcp.Rs.shape[0]):
        R_i = hcp.Rs[i]
        for j in range(i, hcp.Rs.shape[0]):
            R_j = hcp.Rs[j]
            misorientations.append(pyebsd.misorientation(R_i, R_j, out="deg"))

    misorientations = np.array(misorientations)
    np.save("hcp_misorientations", misorientations)
    import pdb;pdb.set_trace()
