import random
import operator
from math import ceil
import sys
from extensions import flatten, triangleReduction
from graph import *

################################################################################
# GRAPH EXTENSION
################################################################################

class Graph(Graph):
    def dna(self): # get a graph's DNA
        return [self.hasEdge(i,j) for i in range(self.nodes) for j in range(self.nodes) if i < j]

    # DNA BIT FORMAT:
    # The new format is a boolean bit which builds consecutively starting from the zero node.

    # EXAMPLE: Graph of size 3 which has the edge list [(0,1), (1,2)]
    # [True, False, True] <-> [(0,1), (0,2), (1,2)]


    def fromDna(self, dna): # birth a graph from DNA
        t = triangleReduction(len(dna))
        if t:
            return Graph(dnaGenerator(dna), t+1)
        else:
            raise Exception("wrong DNA length - must be a triangle number")

def dnaGenerator(dna):
    return lambda r, c: dna[int(r*(r+1)/2+c)]

def generatePopulation(startGraph, graphSize, populationSize):
    """From a prior Ramsey graph, builds a new Ramsey graph"""
    return [Graph(startGraph.generator, graphSize) for x in range(populationSize)]

################################################################################
# TRANSFORMATIONS
################################################################################

def pointInversion(dna):
    """randomly flips a single dna nucleotide"""
    i = random.randint(0, len(dna)-1) # choose random point
    dna[i] = not dna[i] # invert point
    return dna

def crossover(mom, dad):
    """performs random dna crossover similar to actual dna crossover"""
    if len(mom) != len(dad):
        raise Exception("length of DNA are not same between parents")
    crossovers = int(random.betavariate(2, 5) * len(mom)) # gets a number that's more heavily weighted to only a couple of crossovers, not too many, not too few
    # print(crossovers, "crossovers")
    crossoverPoints = random.choices(list(range(0, len(mom))), k=crossovers)
    # print(crossoverPoints, "points")
    
    which = random.randint(0, 1)
    baby = []
    for i in range(0, len(mom)):
        if i in crossoverPoints:
            which = 1 - which # switch parent it's pulling from
        
        if which == 0:
            dna = mom
        else:
            dna = dad
        
        baby.append(dna[i])
    
    return baby

################################################################################
# EVOLUTION
################################################################################

def evaluateGeneration(population, fitnessFunction):
    return [(fitnessFunction(member), member) for member in population]

def evolveByRandomMutation(): # Bacteria!
    """evolves a population by randomly mutating its members"""
    pass

def evolveByRandomSexualReproduction(): # Sponges!
    """evolves a population by randomly mating its members"""
    pass
    
def evolveByRankedSexualReproduction(initialPopulation, fitnessFunction, maxIterations=-1): # Monkeys!
    """evolves a population by mating members with adjacent fitness"""
    population = initialPopulation
    if maxIterations == -1:
        maxIterations = sys.maxsize
    
    for i in range(maxIterations):
        generation = evaluateGeneration(population, fitnessFunction) # FIXME: do not hardcode numbers (except perhaps referencing them from somewhere else)
        bestFitness = min(map(lambda x: x[0], generation))
        generation.sort(key=lambda x: x[0])
        generation = list(map(lambda x: x[1], generation))
        
        # show a marker
        print("Generation: {0} -- best Fitness of: {1}".format(i, bestFitness))
        
        # test if has found member
        if bestFitness == 0:
            return generation[0]
        
        # mate with adjacent member
        newGen = [crossover(generation[i], generation[i+1]) for i in range(len(generation)-1)]
        newGen = list(map(lambda x: pointInversion(x) if random.random() > 1-0.3 else x, newGen)) # point mutate with a 30% chance
        newGen = generation[:1] + newGen
    
    # if reached end of iterations and hasn't found perfect member
    return generation[0]

def evolveByRankedSexualReproductionWithCarryOverAndRandomMutations(initialPopulation, fitnessFunction, maxIterations=-1): # Birds!
    """evolves a population by mating members with a similar fitness and keeping the best members from the previous generation"""
    population = initialPopulation
    if maxIterations == -1:
        maxIterations = sys.maxsize
    
    for i in range(maxIterations):
        generation = evaluateGeneration(population, fitnessFunction) # FIXME: do not hardcode numbers (except perhaps referencing them from somewhere else)
        generation.sort(key=lambda x: x[0])
        
        # keep first 10% of members
        best = gen[ceil(0.05 * len(gen))]
        
        # TODO:
        # mate first one with a couple
        # TODO: use combinations to mate the first and second, third, fifth, eight, etc.
        #       then the second and fourth, ninth, eleventh, etc.
        # mate next ones once with neighbors around them
        # until population size is the same
        
        # print & iterate

################################################################################
# BEES
################################################################################

# Based on the symmetric heuristic, we target the "max graph" in order to reduce that so that the max graph and the min graph
# are roughly equivalent.
def searchAndDestroy(graph, cliqueSize):
    """Selects a random clique of a size, toggles edges from that, and then checks the fitness. Returns None if there
    was no improvement"""
    """ISSUE: it's very slow right now. Generally, if it doesn't find the clique right away, it won't find it at all."""
    graph.getMax()
    bestFitness = graph.fitness(cliqueSize)
    list_of_cliques = graph.findCliques(cliqueSize)[0]
    for i in list_of_cliques:
        copygraph = graph.deepcopy()
        copygraph.toggleClique(i)
        if copygraph.fitness(cliqueSize) < bestFitness:
            return copygraph
    return None

def bruteForce(graph, cliqueSize):
    """Toggles all edges, checks which one gave the lowest fitness, returns that"""
    graph.getMax()
    bestFitness = graph.fitness(cliqueSize)
    bestGraph = graph.deepcopy()
    for i in graph.edgeList():
        graphc = graph.deepcopy()
        graphc.toggleEdge(i[0], i[1])
        if graphc.fitness(cliqueSize) < bestFitness:
            bestGraph = graphc
            bestFitness = bestGraph.fitness(cliqueSize)
    for i in graph.complement().edgeList():
        graphc = graph.deepcopy()
        graphc.toggleEdge(i[0], i[1])
        if graphc.fitness(cliqueSize) < bestFitness:
            bestGraph = graphc
            bestFitness = bestGraph.fitness(cliqueSize)
    return bestGraph

# Change these functions so that they all use numOfBees, and also buildup?

def workerBee(tgraph, cliqueSize, numOfBees):
    """Takes the best graph, toggles edges randomly, sad and brute forces, finds the best fitness among them"""
    dic = {}
    for i in range(numOfBees):
        graph = tgraph.deepcopy()
        graph.toggleRandomEdge()

        isTrue = True
        # Search and destroys until it can't go anymore
        while isTrue:
            graphc = searchAndDestroy(graph, cliqueSize)
            if graphc == None:
                isTrue = False
            elif graphc.fitness(cliqueSize) == graph.fitness(cliqueSize):
                isTrue = False
            else:
                graph = graphc
            try:
                graphc.fitness(cliqueSize)
            except:
                isTrue = False

        # Brute forces until it can't go anymore
        isTrue = True
        while isTrue:
            graphc = bruteForce(graph, cliqueSize)
            if graphc.fitness(cliqueSize) == graph.fitness(cliqueSize):
                isTrue = False
            else:
                graph = graphc

        dic[graph] = graph.fitness(cliqueSize)
    return min(dic, key=dic.get)

def scoutBee(graphSize, cliqueSize, numOfBees):
    """Finds a new graph, and brings it down to as low as it can go"""
    dic = {}
    for i in range(numOfBees):
        graph = Graph(randomGenerator, graphSize)
        isTrue = True
        #Search and destroys until it can't go anymore
        while isTrue:
            graphc = searchAndDestroy(graph, cliqueSize)
            if graphc == None:
                isTrue = False
            elif graphc.fitness(cliqueSize) == graph.fitness(cliqueSize):
                isTrue = False
            else:
                graph = graphc
            try:
                graphc.fitness(cliqueSize)
            except:
                isTrue = False

        #Brute forces until it can't go anymore
        isTrue = True
        while isTrue:
            graphc = bruteForce(graph, cliqueSize)
            if graphc.fitness(cliqueSize) == graph.fitness(cliqueSize):
                isTrue = False
            else:
                graph = graphc
        dic[graph] = graph.fitness(cliqueSize)
    return min(dic, key=dic.get)

def lazyBee(graph, cliqueSize, numOfBees):
    """Takes the best graph, toggles an edge, then runs bruteforce until it can't go any further. Returns the same graph if it 
    doesn't find a better edge."""
    bestGraph = graph
    bestFitness = graph.fitness(cliqueSize)
    count = 0
    isTrue = True
    while isTrue:
        count +=1
        isTrue = False
        for i in range(numOfBees):
            graphc = graph.deepcopy()
            graphc.toggleRandomEdge()
            graphc = bruteForce(graphc, cliqueSize)
            if graphc.fitness(cliqueSize) < bestFitness:
                isTrue = True
                bestGraph = graphc
                bestFitness = graphc.fitness(cliqueSize)
    return bestGraph

def beeMethod(populationSize, numberOfRuns, cliqueSize, graph):
    import math
    # Initialize the bestGraph and bestErr variables.
    best_graph = graph
    best_err = best_graph.fitness(cliqueSize)
    sizeOfGraph = graph.nodes
    print('Start error: ' + str(best_err))

    # Check to see if the fitness is 0
    if best_err == 0:
        return best_graph

    # Trying to find the sweet spot for the population
    numWorker = math.floor(populationSize * 0.44)
    numScout = math.floor(populationSize * 0.22)
    numLazy = math.floor(populationSize * 0.44)
    count = 0

    print("Number of worker bees: " + str(numWorker))
    print("Number of scout bees: " + str(numScout))
    print("Number of lazy bees: " + str(numLazy))

    # Main loop
    for i in range(numberOfRuns):
        tbest_err = best_err
        print("Iteration: " + str(i))

        print('Worker bees')
        # Run worker bees
        newGraph = workerBee(best_graph, cliqueSize, numWorker)
        if newGraph.fitness(cliqueSize) < best_err:
            best_graph = newGraph
            best_err = newGraph.fitness(cliqueSize)

        # Check to see if the fitness is 0
        if best_err == 0:
            print('Found winner')
            return best_graph

        print('Best error so far: ' + str(best_err))
        print('Scout bees')
        # Run scout bees
        newGraph = scoutBee(sizeOfGraph, cliqueSize, numScout)
        if newGraph.fitness(cliqueSize) < best_err:
            best_graph = newGraph
            best_err = newGraph.fitness(cliqueSize)

        # Check to see if the fitness is 0
        if best_err == 0:
            print('Found winner')
            return best_graph

        print('Best error so far: ' + str(best_err))
        print('Lazy bees')
        # Run lazy bees
        new_graph = lazyBee(best_graph, cliqueSize, numLazy)
        if new_graph.fitness(cliqueSize) < best_err:
            best_graph = new_graph
            best_err = new_graph.fitness(cliqueSize)

        # Check to see if the fitness is 0
        if best_err == 0:
            print('Found winner')
            return best_graph

        print('Best error so far: ' + str(best_err))
        if tbest_err == best_err:
            count +=1
        if count > 5:
            print('switching')
            pop = generatePopulation(best_graph, sizeOfGraph, populationSize)
            ff = lambda x: Graph.fromDna(x).fitness(cliqueSize)
            a = evolveByRankedSexualReproduction(list(map(lambda m: m.dna(), pop)), ff, numberOfRuns)
            g = Graph.fromDna(a)
            if g.fitness(cliqueSize) < best_err:
                best_graph = g
                best_err = g.fitness(cliqueSize)
                count = 0
            else:
                count = 0
    print('Best error: ' + str(best_err))
    return best_graph

def buildUpBees(populationSize, numberOfRuns, cliqueSize, startSize, endSize):
    """Starting from the given start size, keep progressively building Ramsey graphs until you reach the endSize"""
    size = startSize
    best_graph = beeMethod(populationSize, numberOfRuns, cliqueSize, Graph(randomGenerator, size))
    while size <= endSize:
        print('Current size: ' + str(size))
        best_graph = beeMethod(populationSize, numberOfRuns, cliqueSize, best_graph)
        if best_graph.fitness(cliqueSize) != 0:
            print('Failed on ' + str(size))
            return best_graph
        size +=1
        best_graph = Graph(best_graph.generator, size)
    return best_graph

#***********************************************************************************************************************
# All was done using graphs, now switching to DNA
def getGraphSize(DNA):
    return triangleReduction(len(DNA)) + 1