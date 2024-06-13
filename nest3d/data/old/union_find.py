class UnionFind:
    def __init__(self, elements):
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}

    def find(self, elem):
        if self.parent[elem] != elem:
            self.parent[elem] = self.find(self.parent[elem])
        return self.parent[elem]

    def union(self, elem1, elem2):
        root1 = self.find(elem1)
        root2 = self.find(elem2)
        
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

def merge_sets(sets):
    # Collect all unique elements
    elements = set()
    for s in sets:
        elements.update(s)
    
    # Initialize Union-Find
    uf = UnionFind(elements)
    
    # Union all elements within each set
    for s in sets:
        lst = list(s)
        for i in range(1, len(lst)):
            uf.union(lst[0], lst[i])
    
    # Group elements by their root
    merged = {}
    for elem in elements:
        root = uf.find(elem)
        if root not in merged:
            merged[root] = set()
        merged[root].add(elem)
    
    # Extract the merged sets
    return list(merged.values())
