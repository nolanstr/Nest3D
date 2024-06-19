import numpy as np
from torch.utils.data import DataLoader

class CustomDataLoader:

    def __init__(self, datasets, keys, batch_size, shuffle):
        self.dataLoaders = [DataLoader(d, batch_size=batch_size,
                                       shuffle=shuffle) for d in datasets]
        self.idx_from_key = dict(zip(keys, np.arange(len(keys)).astype(int))) 
        self.checks = np.array([False for _ in range(len(keys))])

    def __iter__(self):
        self.iterators = [iter(dl) for dl in self.dataLoaders]
        return self

    def __next__(self):
            """
            Returns the next batch from each DataLoader as a tuple.
            """
            if not np.all(self.checks):
                batches = []
                for i, iterator in enumerate(self.iterators):
                    try:
                        batch = next(iterator)
                        batches.append(batch)
                    except StopIteration:
                        # Restart iterator if it reaches the end
                        iterator = iter(self.dataLoaders[i])
                        self.iterators[i] = iterator
                        batch = next(iterator)
                        batches.append(batch)
                        self.checks[i] = True
                
                return tuple(batches)
            else:
                self.checks[:] = False
                raise StopIteration

