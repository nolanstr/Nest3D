import numpy as np
import torch

class Slicer:
    
    def __init__(self, sizes=[49,99,149,199]):
        """
        Parameters
        ----------
        self : object [Argument]
        sizes :default: 49, 99, 149, 199 [Argument]

        """

        self.sizes = sizes
    
    def __call__(self, x: torch.tensor, planes=[2, 1], samples=10):
        """
        Parameters
        ----------
        self : object [Argument]
        x : Volume output from generator with dims (x,y,z) [Argument]
        planes : List of the normal indexes to the planes that are sliced over. 
        samples : Int (number of sub slices to sample)
        """
        return self.generate_slices(x, planes, samples)

    def generate_slices(self, x: torch.tensor, planes=[2, 1], samples=10):
        """
        Parameters
        ----------
        self : object [Argument]
        x : Volume output from generator with dims (x,y,z) [Argument]
        planes : List of the normal indexes to the planes that are sliced over. 
        samples : Int (number of sub slices to sample)
        """
        xShape = x.shape
        x = x.squeeze()
        if min(x.shape) < max(self.sizes):
            raise ValueError("x dims must be >= max(sizes) = {max(self.sizes)}")
        slices = self.initialize_slices(planes)
        dims = x.shape[0] # Assuming cube shape
        for plane in planes:
            
            dims = x.shape[plane]
            for size in self.sizes:
                i, j, k = self.get_index_samples(dims, size, plane, samples)
                sub_slices = []
                for m in range(samples):
                    if plane == 0:
                        sub_slices.append(x[i[m], j[m]:j[m]+size,
                                            k[m]:k[m]+size][None,:,:])
                    if plane == 1:
                        sub_slices.append(x[i[m]:i[m]+size, j[m],
                                            k[m]:k[m]+size][None,:,:])
                    elif plane == 2:
                        sub_slices.append(x[i[m]:i[m]+size, j[m]:j[m]+size,
                                            k[m]][None,:,:])
                    else:
                        raise ValueError("Planes must be combination of (0,1,2)!")
                slices[plane][size] = torch.vstack(sub_slices)

        return slices
    
    def initialize_slices(self, planes):
        slices = {}
        for plane in planes:
            slices[plane] = dict(zip(np.array(self.sizes).astype(int), 
                                                [[] for _ in self.sizes]))
        return slices
    
    def get_index_samples(self, dim, size, plane, samples):
        dim_size_check = False
        if (dim-size) == 0:
            dim_size_check = True
        dimIds = np.arange(dim)
        reducedDimIds = np.arange(dim-size)

        if plane == 0:
            i = np.random.choice(dimIds,
                                 size=samples,
                                 replace=False)
            if dim_size_check:
                j = np.zeros_like(i)
                k = np.zeros_like(i)
            else:
                j = np.random.choice(reducedDimIds,
                                      size=samples,
                                      replace=False)
                k = np.random.choice(reducedDimIds,
                                     size=samples,
                                     replace=False)
        if plane == 1:
            j = np.random.choice(dimIds,
                                  size=samples,
                                  replace=False)
            if dim_size_check:
                i = np.zeros_like(j)
                k = np.zeros_like(j)
            else:
                i = np.random.choice(reducedDimIds,
                                     size=samples,
                                     replace=False)
                k = np.random.choice(reducedDimIds,
                                     size=samples,
                                     replace=False)
        if plane == 2:
            k = np.random.choice(dimIds,
                                 size=samples,
                                 replace=False)
            if dim_size_check:
                i = np.zeros_like(k)
                j = np.zeros_like(k)
            else:
                i = np.random.choice(reducedDimIds,
                                     size=samples,
                                     replace=False)
                j = np.random.choice(reducedDimIds,
                                      size=samples,
                                      replace=False)
        return i, j, k
