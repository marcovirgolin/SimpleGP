# libraries
import numpy as np 
from numpy.random import randint
from numpy.random import random
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from copy import deepcopy

# internal imports
from Nodes.Nodes import *
from Fitness.FitnessFunction import SymbolicRegressionFitness
from Variation import Variation
from Selection import Selection


np.random.seed(42)


X, y = sklearn.datasets.load_boston( return_X_y=True )
#temp = np.copy(X[:,3])
#X[:,3] = y
#y = temp
X = scale(X)
y = scale(y)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42 )

fitness = SymbolicRegressionFitness( X_train, y_train )
functions = [AddNode(), SubNode(), MulNode(), DivNode(), ExpNode(), LogNode(), SinNode(), CosNode()]
functions = [AddNode(), SubNode(), MulNode(), DivNode()]
terminals = [EphemeralRandomConstantNode()]
terminals = []
for i in range(X.shape[1]):
	terminals.append(FeatureNode(i))

crossover_rate = 0.1
mutation_rate = 0.9
max_tree_size = 100

population = []
for i in range(250):
	population.append( Variation.GenerateRandomTree( functions, terminals, max_height=2 ) )
	fitness.Evaluate(population[i])

for g in range(100):

	population = Selection.TournamentSelect( population, len(population), tournament_size=4 )

	O = []
	for i in range(len(population)):
		
		o = deepcopy(population[i])
		if ( random() < crossover_rate ):
			o = Variation.SubtreeCrossover( o, population[randint(len(population))] )
		if ( random() < mutation_rate ):
			o = Variation.SubtreeMutation( o, functions, terminals )
		
		if len(o.GetSubtree()) > max_tree_size:
			del o
			o = deepcopy( population[i])
			# we are copying also the fitness value, no need to re-evaluate
		else:
			fitness.Evaluate(o)

		O.append(o)

	population = O

	print ('g:',g+1,'best fit:', fitness.elite.fitness, 'best size:', len(fitness.elite.GetSubtree()))


print ('Elite training MSE:', np.round(fitness.elite.fitness,3), 
	'Rsquared:', np.round(1.0 - fitness.elite.fitness / np.var(y_train),3))
test_fitness = SymbolicRegressionFitness( X_test, y_test )
test_fitness.Evaluate( fitness.elite )
print ('Elite test MSE:', np.round(test_fitness.elite.fitness,3), 
	'Rsquared:', np.round(1.0 - fitness.elite.fitness / np.var(y_test),3))