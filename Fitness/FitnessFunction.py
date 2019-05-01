import numpy as np
from copy import deepcopy

class SymbolicRegressionFitness:

	def __init__( self, X_train, y_train ):
		self.X_train = X_train
		self.y_train = y_train
		self.elite = None

	def Evaluate( self, individual ):
		output = individual.GetOutput( self.X_train )

		# linear scaling
		try:
			b = np.cov( output, self.y_train )[0,0] / np.var( output )
		except:
			b = 0
		a = np.mean(self.y_train) - np.mean(output)*b

		mean_squared_error = np.mean ( np.square( self.y_train - (a + b*output) ) )
		individual.fitness = mean_squared_error

		if not self.elite or individual.fitness < self.elite.fitness:
			del self.elite
			self.elite = deepcopy(individual)