import numpy as np

from simplegp.Nodes.BaseNode import Node


class AddNode(Node):

    def __init__(self):
        super(AddNode, self).__init__()
        self.arity = 2

    def __repr__(self):
        return '+'

    def get_output(self, X):
        X0 = self.get_child_output(0, X)
        X1 = self.get_child_output(1, X)
        return X0 + X1


class SubNode(Node):
    def __init__(self):
        super(SubNode, self).__init__()
        self.arity = 2

    def __repr__(self):
        return '-'

    def get_output(self, X):
        X0 = self.get_child_output(0, X)
        X1 = self.get_child_output(1, X)
        return X0 - X1


class MulNode(Node):
    def __init__(self):
        super(MulNode, self).__init__()
        self.arity = 2

    def __repr__(self):
        return '*'

    def get_output(self, X):
        X0 = self.get_child_output(0, X)
        X1 = self.get_child_output(1, X)
        return np.multiply(X0, X1)


class DivNode(Node):
    def __init__(self):
        super(DivNode, self).__init__()
        self.arity = 2

    def __repr__(self):
        return '/'

    def get_output(self, X):
        X0 = self.get_child_output(0, X)
        X1 = self.get_child_output(1, X)
        return np.multiply(np.sign(X1), X0) / (1e-2 + np.abs(X1))


class AnalyticQuotientNode(Node):
    def __init__(self):
        super(AnalyticQuotientNode, self).__init__()
        self.arity = 2

    def __repr__(self):
        return 'aq'

    def get_output(self, X):
        X0 = self.get_child_output(0, X)
        X1 = self.get_child_output(1, X)
        return X0 / np.sqrt(1 + np.square(X1))


class ExpNode(Node):
    def __init__(self):
        super(ExpNode, self).__init__()
        self.arity = 1

    def __repr__(self):
        return 'exp'

    def get_output(self, X):
        X0 = self.get_child_output(0, X)
        return np.exp(X0)


class LogNode(Node):
    def __init__(self):
        super(LogNode, self).__init__()
        self.arity = 1

    def __repr__(self):
        return 'log'

    def get_output(self, X):
        X0 = self.get_child_output(0, X)
        return np.log(np.abs(X0) + 1e-2)


class SinNode(Node):
    def __init__(self):
        super(SinNode, self).__init__()
        self.arity = 1

    def __repr__(self):
        return 'sin'

    def get_output(self, X):
        X0 = self.get_child_output(0, X)
        return np.sin(X0)


class CosNode(Node):
    def __init__(self):
        super(CosNode, self).__init__()
        self.arity = 1

    def __repr__(self):
        return 'cos'

    def get_output(self, X):
        X0 = self.get_child_output(0, X)
        return np.cos(X0)


class FeatureNode(Node):
    def __init__(self, id):
        super(FeatureNode, self).__init__()
        self.id = id

    def __repr__(self):
        return 'x' + str(self.id)

    def get_output(self, X):
        return X[:, self.id]


class EphemeralRandomConstantNode(Node):
    def __init__(self):
        super(EphemeralRandomConstantNode, self).__init__()
        self.c = np.nan

    def __instantiate(self):
        self.c = np.round(np.random.random() * 10 - 5, 3)

    def __repr__(self):
        if np.isnan(self.c):
            self.__instantiate()
        return str(self.c)

    def get_output(self, X):
        if np.isnan(self.c):
            self.__instantiate()
        return np.array([self.c] * X.shape[0])
