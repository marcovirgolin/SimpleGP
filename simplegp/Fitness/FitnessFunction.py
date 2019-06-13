from copy import deepcopy

import numpy as np


class SymbolicRegressionFitness:

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.elite = None
        self.evaluations = 0

    def evaluate(self, individual):
        self.evaluations = self.evaluations + 1

        output = individual.get_output(self.X_train)

        mean_squared_error = np.mean(np.square(self.y_train - output))
        individual.fitness = mean_squared_error

        if not self.elite or individual.fitness < self.elite.fitness:
            del self.elite
            self.elite = deepcopy(individual)
