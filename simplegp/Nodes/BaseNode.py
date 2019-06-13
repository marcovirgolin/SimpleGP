import numpy as np


class Node:  # Base class with general functionalities

    def __init__(self):
        self.fitness = np.inf
        self.parent = None
        self.arity = 0  # arity is the number of expected inputs
        self._children = []
        self._children_scaling = []
        self._children_translation = []

    def get_subtree(self):
        result = []
        self.__get_subtree_recursive(result)
        return result

    def append_child(self, N, scale=1, translate=0):
        self._children.append(N)
        self._children_scaling.append(scale)
        self._children_translation.append(translate)
        N.parent = self

    def detach_child(self, N):
        assert (N in self._children)
        for i, c in enumerate(self._children):
            if c == N:
                self._children.pop(i)
                s = self._children_scaling.pop(i)
                t = self._children_translation.pop(i)
                N.parent = None
                return i, s, t

    def insert_child_at_position(self, i, N, scale=1, translate=0):
        self._children.insert(i, N)
        self._children_scaling.insert(i, scale)
        self._children_translation.insert(i, translate)
        N.parent = self

    def get_output(self, X):
        return None

    def get_child_output(self, i, X):
        X0 = self._children[i].get_output(X)
        X0 *= self._children_scaling[i]
        X0 += self._children_translation[i]
        return X0

    def get_depth(self):
        n = self
        d = 0
        while n.parent:
            d = d + 1
            n = n.parent
        return d

    def __get_subtree_recursive(self, result):
        result.append(self)
        for c in self._children:
            c.__get_subtree_recursive(result)
        return result
