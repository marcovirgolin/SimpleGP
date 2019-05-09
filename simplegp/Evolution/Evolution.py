import numpy as np
from numpy.random import random, randint
import time
from copy import deepcopy

from simplegp.Variation import Variation
from simplegp.Selection import Selection


class SimpleGP:

	def __init__(
		self,
		fitness_function,
		functions,
		terminals,
		pop_size=500,
		crossover_rate=0.5,
		mutation_rate=0.5,
		max_evaluations=-1,
		max_generations=-1,
		max_time=-1,
		initialization_max_tree_height=4,
		max_tree_size=100,
		tournament_size=4
		):

		self.pop_size = pop_size
		self.fitness_function = fitness_function
		self.functions = functions
		self.terminals = terminals
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate

		self.max_evaluations = max_evaluations
		self.max_generations = max_generations
		self.max_time = max_time

		self.initialization_max_tree_height = initialization_max_tree_height
		self.max_tree_size = max_tree_size
		self.tournament_size = tournament_size

		self.generations = 0


	def __ShouldTerminate(self):
		must_terminate = False
		elapsed_time = time.time() - self.start_time
		if self.max_evaluations > 0 and self.fitness_function.evaluations >= self.max_evaluations:
			must_terminate = True
		elif self.max_generations > 0 and self.generations >= self.max_generations:
			must_terminate = True
		elif self.max_time > 0 and elapsed_time >= self.max_time:
			must_terminate = True

		if must_terminate:
			print('Terminating at\n\t', 
				self.generations, 'generations\n\t', self.fitness_function.evaluations, 'evaluations\n\t', np.round(elapsed_time,2), 'seconds')

		return must_terminate


	def Run(self):

		self.start_time = time.time()

		population = []
		for i in range( self.pop_size ):
			population.append( Variation.GenerateRandomTree( self.functions, self.terminals, self.initialization_max_tree_height ) )
			self.fitness_function.Evaluate( population[i] )

		while not self.__ShouldTerminate():

			O = []
			
			for i in range(len(population) - 1):
				
				o = deepcopy(population[i])
				if ( random() < self.crossover_rate ):
					o = Variation.SubtreeCrossover( o, population[randint(len(population))] )
				if ( random() < self.mutation_rate ):
					o = Variation.SubtreeMutation( o, self.functions, self.terminals, max_height=self.initialization_max_tree_height )
				
				if len(o.GetSubtree()) > self.max_tree_size:
					del o
					o = deepcopy( population[i] )
				else:
					self.fitness_function.Evaluate(o)

				O.append(o)

			PO = population+O
			population = Selection.TournamentSelect( PO, len(population), tournament_size=self.tournament_size )

			self.generations = self.generations + 1

			print ('g:',self.generations,'elite fitness:', np.round(self.fitness_function.elite.fitness,3), ', size:', len(self.fitness_function.elite.GetSubtree()))
