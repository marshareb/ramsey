import sys
import time

from graph import *
from evolution import *

# NOTE: We only care about what are called diagonal Ramsey numbers (since R(5,5)
# is our end goal).
################################################################################
# TESTING
################################################################################

def ramseyTest(population, numberOfRuns, cliqueSize, size, fitness):
    """Basic algorithm to find counterexamples. It's very inefficient."""
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
        pop[a] = fitness(a, cliqueSize)
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
            k.toggleRandomEdge()
            pop[k] = fitness(k, cliqueSize)
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
                pop[a] = fitness(a, cliqueSize)

        # That way, we can keep track of if it's running and how long each iteration is taking
        print("Iteration: {0} - Best Fitness: {1} - worstFitness {2}".format(run, bestFitness, worstFitness))

        # Reset worstFitness
        worstFitness = 0

    # Once we've cycled through all of the numberOfRuns, we conclude by saying how close we got to our goal of 0.
    return bestGraph

def graph():
    """Example of a graph."""
    a = Graph(randomGenerator, 6)
    print(a)
    print(a.findCliques(3))
    print(a.fitness(3))
    a.draw()

def dnaGenerator():
    """Example of using a DNA bit to generate a graph."""
    a = Graph(dnaGenerator([True, True, True, True, True, True]), 4)
    print(a)
    print(a.fitness(3))
    a.draw()

def Ramsey(populationSize, numberOfRuns, cliqueSize, size):
    """Example of Ramsey test algorithm."""
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

def monkeyEvolve(popSize, iterations, cliqueSize, graphSize):
    """Example of monkey evolution algorithm."""
    pop = [Graph(randomGenerator, graphSize) for x in range(popSize)]
    ff = lambda x: fromDna(x).fitness(cliqueSize)
    a = evolveByRankedSexualReproduction(list(map(lambda m: m.dna(), pop)), ff, iterations)
    g = fromDna(a)
    cs = g.findCliques(cliqueSize)
    print("{0}-Cliques: {1}".format(cliqueSize, cs[0]))
    print("{0}-Anti-Cliques: {1}".format(cliqueSize, cs[1]))
    print(g)
    g.draw()
    g.draw2()


def fullSizeGraph():
    """Example of drawing a graph."""
    a = Graph(randomGenerator, 43)
    a.draw()
    a.draw2()

def standardBees():
    """Example of using the bee algorithm with the standard fitness."""
    start = time.time()
    x= buildUpBees(1000, 800, 100, 4, 12, 17, fitness)
    print("Time elapsed: " + str(time.time() - start))
    x.draw2()
    if fitness(x, 4) == 0:
        x.write_to_file('counterexample_17_4.txt')
    x.draw2()

def symmetricBees():
    """Example of using the bee algorithm with the symmetric fitness."""
    start = time.time()
    x= buildUpBees(1000, 800, 100, 4, 12, 17, symmetricFitness)
    print("Time elapsed: " + str(time.time() - start))
    x.draw2()
    if fitness(x, 4) == 0:
        x.write_to_file('counterexample_17_4.txt')
    x.draw2()

if __name__ == "__main__":
    print("Running through examples.")
    print("Note that they might not find counterexamples since they rely on elements of randomness.")
    print("The goal of these algorithms is to try and find a counterexample for R(4,4) at size 17.")

    print("Generic Ramsey Test...")
    x = ramseyTest([Graph(randomGenerator, 17) for i in range(500)], 20, 4, 17, fitness)
    print("Best error: " + str(fitness(x,4)))

    print("Initializing simulated annealing...")

    print("Symmetric ordered simulated annealing...")
    x = simulatedAnnealing(500, 17, 4, symmetricFitness)
    print("Best error: " + str(fitness(x,4)))

    print("Standard ordered simulated annealing...")
    x = simulatedAnnealing(500, 17, 4, fitness)
    print("Best error: " + str(fitness(x, 4)))

    print("Symmetric random simulated annealing...")
    x = simulatedAnnealingRandom(250, 17, 4, symmetricFitness)
    print("Best error: " + str(fitness(x,4)))

    print("Standard random simulated annealing...")
    x = simulatedAnnealingRandom(250, 17, 4, fitness)
    print("Best error: " + str(fitness(x,4)))


    print("Initializing swarm algorithms...")

    print("Symmetric bees...")
    symmetricBees()

    print("Standard bees...")
    standardBees()

    print("Initializing genetic algorithms...")

    print("Monkey evolution...")
    x = monkeyEvolve(100, 100, 4, 17)
    print("Best error: " + str(fitness(x,4)))
