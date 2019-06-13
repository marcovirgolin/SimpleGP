# Libraries
import sklearn.datasets
from sklearn.model_selection import train_test_split

from simplegp.Evolution.Evolution import SimpleGP
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
# Internal imports
from simplegp.Nodes.SymbolicRegressionNodes import *

np.random.seed(42)

# Load regression dataset 
X, y = sklearn.datasets.load_boston(return_X_y=True)
# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Set fitness function
fitness_function = SymbolicRegressionFitness(X_train, y_train)

# Set functions and terminals
functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node
for i in range(X.shape[1]):
    terminals.append(FeatureNode(i))  # add a feature node for each feature

# Run GP
sgp = SimpleGP(fitness_function, functions, terminals, pop_size=100,
               max_generations=100)  # other parameters are optional
sgp.run()

# Print results
# Show the evolved function
final_evolved_function = fitness_function.elite
nodes_final_evolved_function = final_evolved_function.get_subtree()
print('Function found (', len(nodes_final_evolved_function), 'nodes ):\n\t',
      nodes_final_evolved_function)  # this is in Polish notation
# Print results for training set
print('Training\n\tMSE:', np.round(final_evolved_function.fitness, 3),
      '\n\tRsquared:', np.round(1.0 - final_evolved_function.fitness / np.var(y_train), 3))
# Re-evaluate the evolved function on the test set
test_prediction = final_evolved_function.get_output(X_test)
test_mse = np.mean(np.square(y_test - test_prediction))
print('Test:\n\tMSE:', np.round(test_mse, 3),
      '\n\tRsquared:', np.round(1.0 - test_mse / np.var(y_test), 3))

# Test get/set scaling
original_scaling = final_evolved_function.get_subtree_scaling()
set_scaling = np.random.uniform(0, 1, len(original_scaling)).tolist()
final_evolved_function.set_subtree_scaling(set_scaling)
get_scaling = final_evolved_function.get_subtree_scaling()
assert set_scaling == get_scaling

# Test get/set translations
original_translation = final_evolved_function.get_subtree_translation()
set_translation = np.random.uniform(-5, +5, len(original_translation)).tolist()
final_evolved_function.set_subtree_translation(set_translation)
get_translation = final_evolved_function.get_subtree_translation()
assert set_translation == get_translation
