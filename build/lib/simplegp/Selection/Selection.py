import numpy as np
from copy import deepcopy
from numpy.random import randint

def TournamentSelect( population, how_many_to_select, tournament_size=4 ):
	# this is a stocastic variation of tournament selection
	pop_size = len(population)
	selection = []

	while len(selection) < how_many_to_select:

		best = population[randint(pop_size)]
		for i in range(tournament_size - 1):
			contestant = population[randint(pop_size)]
			if contestant.fitness < best.fitness:
				best = contestant

		survivor = deepcopy(best)
		selection.append(survivor)

	return selection