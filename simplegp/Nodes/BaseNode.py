import numpy as np


class Node:  # Base class with general functionalities

    def __init__(self):
        self.fitness = np.inf
        self.parent = None
        self.arity = 0  # arity is the number of expected inputs
        self._children = []

    def get_subtree(self):
        result = []
        self.__get_subtree_recursive(result)
        return result

    def append_child(self, N):
        self._children.append(N)
        N.parent = self

    def detach_child(self, N):
        assert (N in self._children)
        for i, c in enumerate(self._children):
            if c == N:
                self._children.pop(i)
                N.parent = None
                return i

    def insert_child_at_position(self, i, N):
        self._children.insert(i, N)
        N.parent = self

    def get_output(self, X):
        return None

    def get_depth(self):
        n = self
        d = 0
        while (n.parent):
            d = d + 1
            n = n.parent
        return d

    def __get_subtree_recursive(self, result):
        result.append(self)
        for c in self._children:
            c.__get_subtree_recursive(result)
        return result
