import sys
import time

from graph import *
from evolution import *

# TODO: Read the summary of http://ieeexplore.ieee.org/document/5461802/. It 
# talks a bit about how we can construct counterexamples from prior 
# counterexamples. Maybe James needs to prove that all Ramsey graphs Gi exist as
# subgraphs of graphs G(i+1)?

# TODO: In the checking for all cliques, we can take a large subset instead of 
# scanning the entire thing, narrowing down all possibilities to a smaller 
# subset to search through.

# TODO: Need to add some way to merge graphs. Merging graphs involves selecting 
# a random number of edges from one graph and a random number of edges from 
# another graph and then putting those edges together (Edges in this case 
# includes the possibility of not having an edge. Essentially grabs a random 
# number of boolean values from one graph and a random number of boolean values 
# from another graph and merges them this way.)

# NOTE: We only care about what are called diagonal Ramsey numbers (since R(5,5)
# is our end goal).
# Let's create a sufficiently random population of graphs. For crossover 
# (or mating) we simply take random values from each graph. If the fitness is 
# better than our worst fitness, we then add this graph to our population
# We should also have an efficient way to output the data to a text file for analysis

################################################################################
# TESTING
################################################################################

def ramseyTest(population, numberOfRuns, cliqueSize, size):
    import sys
    import copy
    # Dictionary to hold all of the different graphs associated to their fitness number
    pop = {}

    # Initialize variables
    bestFitness = sys.maxsize
    bestGraph = Graph(0, 0)
    worstFitness = 0

    # Generate all the graphs, figure out bestFitness and worstFitness
    for a in population:
        pop[a] = a.fitness(cliqueSize)
        if pop[a] < bestFitness:
            bestFitness = pop[a]
            bestGraph = copy.deepcopy(a)
        if pop[a] > worstFitness:
            worstFitness = pop[a]

    # Check to see if we've already finished
    if bestFitness == 0:
        return bestGraph

    # Run through the genetic simulation
    for run in range(numberOfRuns):
        
        # Mutate each graph
        for k in pop:
            k.mutate(cliqueSize)
            pop[k] = k.fitness(cliqueSize)
            if pop[k] < bestFitness:
                bestFitness = pop[k]
                bestGraph = copy.deepcopy(k)
            if pop[k] > worstFitness:
                worstFitness = pop[k]
        # Check to see if we've found a winner
        if bestFitness == 0:
            return bestGraph
        # Remove all of the bad graphs and replace them with new graphs
        for i in pop:
            if pop[i] == worstFitness:
                del pop[i]
                a = Graph(randomGenerator, size)
                pop[a] = a.fitness(cliqueSize)

        # That way, we can keep track of if it's running and how long each iteration is taking
        print("Iteration: {0} - Best Fitness: {1} - worstFitness {2}".format(run, bestFitness, worstFitness))

        # Reset worstFitness
        worstFitness = 0

    # Once we've cycled through all of the numberOfRuns, we conclude by saying how close we got to our goal of 0.
    print(bestFitness)
    return bestGraph

################################################################################
# EXECUTION
################################################################################

'''
def testGraph():
    a = Graph(randomGenerator, 6)
    print(a)
    print(a.findCliques(3))
    print(a.fitness(3))
    a.draw()

def testDnaGenerator():
    a = Graph(dnaGenerator([True, True, True, True, True, True]), 4)
    print(a)
    print(a.fitness(3))
    a.draw()

def testRamsey(populationSize, numberOfRuns, cliqueSize, size):
    pop = [Graph(randomGenerator, 1) for x in range(populationSize)]
    for i in range(1, size+1):
        a = ramseyTest(pop, numberOfRuns, cliqueSize, i)
        print("Fitness: {0}".format(a.fitness(cliqueSize)))
        cs = a.findCliques(cliqueSize)
        print("{0}-Cliques: {1}".format(cliqueSize, cs[0]))
        print("{0}-Anti-Cliques: {1}".format(cliqueSize, cs[1]))
        print(a)
        if a.fitness(cliqueSize) != 0:
            break
        pop = generatePopulation(a, i+1, populationSize)
        print(pop)
    a.draw2()

def testMonkeyEvolve(popSize, iterations, cliqueSize, graphSize):
    pop = [Graph(randomGenerator, graphSize) for x in range(popSize)]
    ff = lambda x: Graph.fromDna(x).fitness(cliqueSize)
    a = evolveByRankedSexualReproduction(list(map(lambda m: m.dna(), pop)), ff, iterations)
    g = Graph.fromDna(a)
    cs = g.findCliques(cliqueSize)
    print("{0}-Cliques: {1}".format(cliqueSize, cs[0]))
    print("{0}-Anti-Cliques: {1}".format(cliqueSize, cs[1]))
    print(g)
    g.draw()
    g.draw2()


def testFullSizeGraph():
    a = Graph(randomGenerator, 43)
    a.draw()
    a.draw2()

'''

def Bees():
    start = time.time()
    x= buildUpBees(1000, 800, 100, 4, 15, 17)
    print("Time elapsed: " + str(time.time() - start))
    x.draw2()
    if x.fitness(4) == 0:
        x.write_to_file('counterexample_17_4.txt')
    x.draw2()

if __name__ == "__main__":
    # An example of the simulated annealing for R(4)
    x = simulatedAnnealing(100, 17, 4)
    # An example of building up the bees for R(4)
    Bees()
    # An example for genetic algorithms
