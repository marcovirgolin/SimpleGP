# Libraries

import random

from cross_validation import CrossValidation
from simplegp.Evolution.Evolution import SimpleGP
# Internal imports
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Weights.Tuner import Tuner

np.random.seed(42)
random.seed(42)

# Set functions and terminals
functions = [AddNode(), SubNode(), MulNode(), AnalyticQuotientNode()]  # chosen function nodes
terminals = [EphemeralRandomConstantNode()]  # use one ephemeral random constant node

# Run GP

tuner = Tuner()
sgp = SimpleGP(tuner=tuner, functions=functions, pop_size=100, max_generations=100)  # other parameters are optional

CrossValidation(sgp, terminals).validate()
