# Libraries
import numpy as np 
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from copy import deepcopy

# Internal imports
from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Evolution.Evolution import SimpleGP

from simplegp.SKLearnInterface import GPSymbolicRegressionEstimator as GPE

np.random.seed(42)

# Load regression dataset 
X, y = sklearn.datasets.load_boston( return_X_y=True )

# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42 )

# Initalize GP estimator
use_linear_scaling=True
gpe = GPE(pop_size=50, max_generations=25, verbose=True, max_tree_size=100, 
	crossover_rate=0.0, mutation_rate=0.33, op_mutation_rate=0.33, 
  min_height=2, initialization_max_tree_height=4, 
	tournament_size=4, max_features=-1, use_linear_scaling=use_linear_scaling,
	functions = [ AddNode(), SubNode(), MulNode(), DivNode() ])

# Fit
gpe.fit(X_train,y_train)

# Get final result
elite_info = gpe.get_elitist_info()
elite = elite_info[0]
elite_str = elite.GetHumanExpression()
if use_linear_scaling:
    linear_scaling_a = elite_info[1]
    linear_scaling_b = elite_info[2]
    elite_str = str(linear_scaling_a) + '+' + str(linear_scaling_b) + '*'+ elite_str
print('Best individual found:', elite_str)

# Show mean squared error
print('Train MSE:',np.mean(np.square(y_train - gpe.predict(X_train))))
print('Test RMSE:',np.mean(np.square(y_test - gpe.predict(X_test))))

# A simple example of retrieving info about the population
population = gpe.get_population()
fraction_unique_individuals = len(np.unique([str(x.GetSubtree()) for x in population])) / len(population)
print('Population convergence:', 100*(1-fraction_unique_individuals), '%')

quit()
# The code below shows how to perform cross-validation
from sklearn.model_selection import cross_validate

cv_result = cross_validate(gpe, X, y, scoring='neg_mean_squared_error', cv=5)

print (cv_result)
