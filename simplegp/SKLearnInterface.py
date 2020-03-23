from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import inspect

from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Evolution.Evolution import SimpleGP


class GPSymbolicRegressionEstimator(BaseEstimator, RegressorMixin):

	def __init__(self, pop_size=100, 
		max_generations=100, 
		max_evaluations=-1,
		max_time=-1,
		functions=[ AddNode(), SubNode(), MulNode(), DivNode() ], 
		use_erc=True,
		crossover_rate=0.5,
		mutation_rate=0.5,
		initialization_max_tree_height=6,
		tournament_size=4,
		max_tree_size=100, max_features=-1, 
		use_linear_scaling=True, verbose=False ):

		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop('self')
		for arg, val in values.items():
			setattr(self, arg, val)


	def fit(self, X, y):

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		self.X_ = X
		self.y_ = y
		
		fitness_function = SymbolicRegressionFitness( X, y, self.use_linear_scaling )
		
		terminals = []
		if self.use_erc:
			terminals.append( EphemeralRandomConstantNode() )
		n_features = X.shape[1]
		for i in range(n_features):
			terminals.append(FeatureNode(i))

		sgp = SimpleGP(fitness_function, self.functions, terminals, 
			pop_size=self.pop_size, 
			max_generations=self.max_generations,
			max_time = self.max_time,
			max_evaluations = self.max_evaluations,
			crossover_rate=self.crossover_rate,
			mutation_rate=self.mutation_rate,
			initialization_max_tree_height=self.initialization_max_tree_height,
			max_tree_size=self.max_tree_size,
			max_features=self.max_features,
			tournament_size=self.tournament_size,
			verbose=self.verbose)

		sgp.Run()

		self.gp_ = sgp

		return self

	def predict(self, X):
		# Check fit has been called
		check_is_fitted(self, ['gp_'])

		# Input validation
		X = check_array(X)

		fifu = self.gp_.fitness_function

		prediction = fifu.elite_scaling_a + fifu.elite_scaling_b * fifu.elite.GetOutput( X )

		return prediction

	def score(self, X, y=None):
		if y is None:
			raise ValueError('The ground truth y was not set')
		
		# Check fit has been called
		prediction = self.predict(X)
		return -1.0 * np.mean(np.square(y - prediction))


	def get_params(self, deep=True):
		attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		attributes = [a for a in attributes if not (a[0].endswith('_') or a[0].startswith('_'))]

		dic = {}
		for a in attributes:
			dic[a[0]] = a[1]

		return dic


	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def get_elitist_info(self):
		check_is_fitted(self, ['gp_'])
		result = ( self.gp_.fitness_function.elite, self.gp_.fitness_function.elite_scaling_a, self.gp_.fitness_function.elite_scaling_b )
		return result