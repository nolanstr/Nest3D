import numpy as np
from defdap.quat import Quat

class Beta:
    
    def __init__(self, ebsdMap, grain_ids, beta_grain_id):
        
        self.ebsdMap = ebsdMap
        self.grain_ids = grain_ids
        self.beta_grain_id = beta_grain_id
        self.alpha_grains = [ebsdMap[gID] for gID in grain_ids] 
        self.alpha_grain_oris = [grain.refOri for grain in self.alpha_grains]
        self.get_all_alpha_grain_oris()
        self.get_parent_beta_oris()
        self.sym_alpha_grain_oris = Quat.calcSymEqvs(
                [grain.refOri for grain in self.alpha_grains], 
                self.alpha_grains[0].crystalSym)
        self.estimate_beta_grain_ori()
        self.update_grain_beta_ids()

    def update_grain_beta_ids(self):
        for grain in self.alpha_grains:
            grain.beta_id = self.beta_grain_id

    def estimate_beta_grain_ori(self, method="maximum", tol=1.0):
        self.check_parent_beta_oris(tol)
        self.refOri = self.parent_beta_oris[np.argmin(self.mean_misOri)]
    
    def check_parent_beta_oris(self, tol=1.0):
        mean_misOri = []
        for parent_beta_ori in self.parent_beta_oris:
            misOris = self.compute_beta_oris_misOri(parent_beta_ori)
            minMisOris = np.min(misOris, axis=0)
            mean_misOri.append(np.mean(np.array(misOris)))
        self.mean_misOri = np.array(mean_misOri)

    def get_parent_beta_oris(self):
        parent_beta_oris = []
        for grain in self.alpha_grains:
            if grain.parent_beta_ori != None:
                parent_beta_oris.append(grain.parent_beta_ori)
            else:
                votes = grain.variant_count
                winners = np.argwhere(votes == np.max(votes)).flatten()
                possible_beta_oris = grain.beta_oris
                for idx in winners:
                    parent_beta_oris.append(possible_beta_oris[idx])

        self.parent_beta_oris = parent_beta_oris

    def get_all_alpha_grain_oris(self):
        
        all_oris = []
        for grain in self.alpha_grains:
            syms = Quat.symEqv(grain.crystalSym)
            refOri = grain.refOri
            for sym in syms:
                all_oris.append((sym*refOri).quatCoef)
        self.all_alpha_grain_oris = np.array(all_oris)
         
    def compute_beta_oris_misOri(self, refOri):
        misOris = np.empty((self.sym_alpha_grain_oris.shape[0],
                            self.sym_alpha_grain_oris.shape[2]))
        misOris[:, :] = abs(np.einsum("ijk,j->ik", 
                                      self.sym_alpha_grain_oris, 
                                      refOri.quatCoef))
        return misOris
