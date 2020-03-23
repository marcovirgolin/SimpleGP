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


gpe = GPE(pop_size=100, max_generations=10, verbose=True, max_tree_size=-1, 
	crossover_rate=0.25, mutation_rate=0.75, initialization_max_tree_height=4, 
	tournament_size=4, max_features=-1,
	functions = [ AddNode(), SubNode(), MulNode(), DivNode() ])

gpe.fit(X_train,y_train)

get_elitist_info = gpe.get_elitist_info()
print(get_elitist_info[0].GetSubtree(), get_elitist_info[1], get_elitist_info[2])

from sklearn.model_selection import cross_validate

cv_result = cross_validate(gpe, X, y, scoring='neg_mean_squared_error', cv=5)

print (cv_result)
