import numpy as np
import pickle
import os
import glob
import matplotlib.pyplot as plt

from sample_squares import *

def generate_images(ebsdMap, out, n=20, sizes=[19,49,99,149,199]):
    x, y = np.arange(450), np.arange(450)
    X, Y = np.meshgrid(x, y)
    coords = np.vstack((X.flatten(), Y.flatten()))
    all_alphas = make_alpha_matrix(ebsdMap)    
    OFFSET = 0
    for beta in ebsdMap.beta_grains:
        R = ebsdMap.beta_grains[0].refOri.rotMatrix()
        rotated_coords = R[:2,:2] @ coords 
        alphas = make_beta_grain_alpha_matrix(beta)
        beta_coords = get_beta_alpha_coords(beta)
        values, indexes = extract_values_and_update_indexes(alphas,
                                                            beta_coords)
        squares = sample_sized_squares(indexes, 
                                 values,
                                 sizes,
                                 num_squares=10)
        if squares:
            for i, square in enumerate(squares):
                tag = f"img{i+OFFSET}"
                save_square(square, out, tag)
            OFFSET += len(squares)

def save_square(square, out, tag):
    n = int(square.shape[0]**0.5)
    x, y = np.arange(n), np.arange(n)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, square.reshape((n,n)), cmap=plt.cm.tab20)
    plt.savefig(f"{out}/pngs/{tag}", dpi=300)
    plt.clf()
    np.save(f"{out}/{tag}", square.reshape((n,n)))

def get_beta_alpha_coords(beta):
    coords = []
    for grain in beta.alpha_grains:
        idxs = np.array(grain.coordList)
        coords.append(idxs)
    return np.vstack(coords)

def make_alpha_matrix(ebsdMap):
    alphas = np.zeros((450,450)) * np.nan
    for beta in ebsdMap.beta_grains:
        for grain in beta.alpha_grains:
            idxs = np.array(grain.coordList)
            alphas[idxs[:,0], idxs[:,1]] = grain.alpha_id 
    return alphas

def make_beta_grain_alpha_matrix(beta):
    alphas = np.zeros((450,450)) * np.nan
    for grain in beta.alpha_grains:
        idxs = np.array(grain.coordList)
        alphas[idxs[:,0], idxs[:,1]] = grain.alpha_id 
    return alphas

if __name__ == "__main__":

    DIRS = glob.glob("./outputs/*")
    for DIR in DIRS:
        f = open(f"{DIR}/final_ebsdMap.pkl", "rb")
        ebsdMap = pickle.load(f)
        f.close()
        out = f"images/{DIR.split('/')[-1]}"
        if not os.path.isdir(out):
            os.system(f"mkdir {out}")
            os.system(f"mkdir {out}/pngs")

        generate_images(ebsdMap, out, n=10)
    import pdb;pdb.set_trace()
