# Libraries
import numpy as np 
import sklearn.datasets
from sklearn.model_selection import train_test_split
from copy import deepcopy

# Internal imports
from Nodes.BaseNode import Node
from Nodes.SymbolicRegressionNodes import *
from Fitness.FitnessFunction import SymbolicRegressionFitness
from Evolution.Evolution import SimpleGP

np.random.seed(42)

# Load regression dataset 
X, y = sklearn.datasets.load_boston( return_X_y=True )
# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42 )
# Set fitness function
fitness_function = SymbolicRegressionFitness( X_train, y_train )

# Set functions and terminals
functions = [ AddNode(), SubNode(), MulNode(), AnalyticQuotientNode() ]	# chosen function nodes	
terminals = [ EphemeralRandomConstantNode() ]	# use one ephemeral random constant node
for i in range(X.shape[1]):
	terminals.append(FeatureNode(i))	# add a feature node for each feature

sgp = SimpleGP(fitness_function, functions, terminals, pop_size=100, max_time=30)	# other parameters are optional
sgp.Run()

# Print results
# Show the evolved function
final_evolved_function = fitness_function.elite
nodes_final_evolved_function = final_evolved_function.GetSubtree()
print ('Function found (',len(nodes_final_evolved_function),'nodes ):\n\t', nodes_final_evolved_function) # this is in Polish notation
# Print results for training set
print ('Training\n\tMSE:', np.round(final_evolved_function.fitness,3), 
	'\n\tRsquared:', np.round(1.0 - final_evolved_function.fitness / np.var(y_train),3))
# Re-evaluate the evolved function on the test set
test_fitness_function = SymbolicRegressionFitness( X_test, y_test )
test_fitness_function.Evaluate( final_evolved_function )
print ('Test:\n\tMSE:', np.round( final_evolved_function.fitness, 3), 
	'\n\tRsquared:', np.round(1.0 - final_evolved_function.fitness / np.var(y_test),3))
