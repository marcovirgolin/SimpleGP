import numpy as np
from copy import deepcopy

class SymbolicRegressionFitness:

	def __init__( self, X_train, y_train ):
		self.X_train = X_train
		self.y_train = y_train
		self.elite = None
		self.evaluations = 0

	def getFitness( self, individual ):

		output = np.arctan(individual.GetOutput( self.X_train ))
		output[output >= 0] = 1
		output[output < 0] = -1
		
		error_rate = 1 - (np.sum(self.y_train == output) / len(self.y_train))
		return error_rate
	
		
	def Evaluate( self, individual ):

		self.evaluations = self.evaluations + 1
		
		error_rate = self.getFitness(individual)
		
		individual.fitness = error_rate

		if not self.elite or individual.fitness < self.elite.fitness:
			del self.elite
			self.elite = deepcopy(individual)
			

			
			