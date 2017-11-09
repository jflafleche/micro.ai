import numpy as np

class SumTree():
    """
    Creates a sum tree.

    Modified from https://github.com/jaara/AI-blog/blob/master/SumTree.py
    and https://morvanzhou.github.io/tutorials/

    Example
        for capacity = 4

                    0
                  /   \
                 1     2
                / \   / \
               3   4 5   6

        tree shape=(2*capacity-1,) stores sums
        data shape=(capacity) stores data
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        """
        Propagates the change value of a leaf up a tree to 
        the top of the tree.
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        Returns index if no children, otherwise returns
        the child with a sum value smaller than the sum 
        being requested.
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx
        
        if self.tree[left] == self.tree[right]:
            return self._retrieve(np.random.choice([left, right]), s)
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data
        self.update(idx, p)

        self.data_pointer += 1
        if self.data_pointer > self.capacity:
            self.data_pointer = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]

    @property
    def root_priority(self):
        return self.tree[0]  # the root