from scipy.io import arff
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Evolution.Evolution import SimpleGP

# Load dataset
data, meta = arff.loadarff('data/cpu_act.arff')
df = pd.DataFrame(data)

# target variable
y = df.binaryClass.values
print(type(y))

# other data
X = df.drop('binaryClass', axis=1).values #df.iloc[:, df.columns != 'binaryClass'].values
print(type(X))


# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print('Type of X_train: ', X_train.dtype)
print('Type of y_train: ', y_train.dtype)

# Set fitness function
fitness_function = SymbolicRegressionFitness(X_train, y_train)
print(fitness_function)


# Set functions and terminals
functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node
for i in range(X.shape[1]):
    terminals.append(FeatureNode(i))  # add a feature node for each feature

# Run GP
sgp = SimpleGP(fitness_function, functions, terminals, pop_size=100, max_generations=100)  # other parameters are optional
sgp.Run()

# Print results
# Show the evolved function
final_evolved_function = fitness_function.elite
nodes_final_evolved_function = final_evolved_function.GetSubtree()
print('Function found (', len(nodes_final_evolved_function), 'nodes ):\n\t',
      nodes_final_evolved_function)  # this is in Polish notation

# Print results for training set
print('Training\n\tMSE:', np.round(final_evolved_function.fitness, 3),
      '\n\tRsquared:', np.round(1.0 - final_evolved_function.fitness / np.var(y_train), 3))

# Re-evaluate the evolved function on the test set
test_prediction = final_evolved_function.GetOutput(X_test)
test_mse = np.mean(np.square(y_test - test_prediction))
print('Test:\n\tMSE:', np.round(test_mse, 3),
      '\n\tRsquared:', np.round(1.0 - test_mse / np.var(y_test), 3))
