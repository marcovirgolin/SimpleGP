from copy import deepcopy

import numpy as np
from numpy.random import randint
from numpy.random import random


def generate_random_tree(functions, terminals, max_height, curr_height=0):
    if curr_height == max_height:
        idx = randint(len(terminals))
        n = deepcopy(terminals[idx])
    else:
        if random() < 0.5:
            n = deepcopy(terminals[randint(len(terminals))])
        else:
            idx = randint(len(functions))
            n = deepcopy(functions[idx])
            for i in range(n.arity):
                c = generate_random_tree(functions, terminals, max_height, curr_height=curr_height + 1)
                n.append_child(c)  # do not use n.children.append because that won't set the n as parent node of c

    return n


def subtree_mutation(individual, functions, terminals, max_height=4):
    mutation_branch = generate_random_tree(functions, terminals, max_height)

    nodes = individual.get_subtree()

    nodes = __get_candidate_nodes_at_uniform_random_depth(nodes)

    to_replace = nodes[randint(len(nodes))]

    if not to_replace.parent:
        del individual
        return mutation_branch

    p = to_replace.parent
    idx = p.detach_child(to_replace)
    p.insert_child_at_position(idx, mutation_branch)

    return individual


def subtree_crossover(individual, donor):
    # this version of crossover returns 1 child

    nodes1 = individual.get_subtree()
    nodes2 = donor.get_subtree()  # no need to deep copy all nodes of parent2

    nodes1 = __get_candidate_nodes_at_uniform_random_depth(nodes1)
    nodes2 = __get_candidate_nodes_at_uniform_random_depth(nodes2)

    to_swap1 = nodes1[randint(len(nodes1))]
    to_swap2 = deepcopy(nodes2[randint(len(nodes2))])  # we deep copy now, only the sutbree from parent2

    p1 = to_swap1.parent

    if not p1:
        return to_swap2

    idx = p1.detach_child(to_swap1)
    p1.insert_child_at_position(idx, to_swap2)

    return individual


def __get_candidate_nodes_at_uniform_random_depth(nodes):
    depths = np.unique([x.get_depth() for x in nodes])
    chosen_depth = depths[randint(len(depths))]
    candidates = [x for x in nodes if x.get_depth() == chosen_depth]

    return candidates
