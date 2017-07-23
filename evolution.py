from math import ceil
import sys
from graph import *
from multiprocessing.dummy import Pool as ThreadPool
# Initialize the number of pools that we will be using. Default is 4.
pool = ThreadPool(6)

################################################################################
# GRAPH EXTENSION
################################################################################

def dnaGenerator(dna):
    """Generates a graph from a dna bit"""
    return lambda r, c: dna[int(r*(r+1)/2+c)]

def generatePopulation(startGraph, graphSize, populationSize):
    """From a prior Ramsey graph, builds a list of new Ramsey Graphs"""
    return [Graph(startGraph.generator, graphSize) for i in range(populationSize)]

################################################################################
# TRANSFORMATIONS
################################################################################

def pointInversion(dna):
    """Randomly flips a single dna nucleotide. Alternative to randomToggle"""
    i = random.randint(0, len(dna)-1) # choose random point
    dna[i] = not dna[i] # invert point
    return dna

def crossover(mom, dad):
    """Performs random dna crossover similar to actual dna crossover."""
    if len(mom) != len(dad):
        raise Exception("Length of DNA are not same between parents")
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

# TODO: Fix these guys
def evaluateGeneration(population, fitnessFunction):
    """Finds the fitness of all the graphs in a list."""
    return [(fitnessFunction(member), member) for member in population]

def evolveByRandomMutation(initialPopulation, fitnessFunction, maxIterations=-1): # Bacteria!
    """Evolves a population by randomly mutating its members"""
    population = initialPopulation
    if maxIterations == -1:
        maxIterations = sys.maxsize

    for i in range(maxIterations):
        generation = evaluateGeneration(population, fitnessFunction)
        bestFitness = min(map(lambda x: x[0], generation))
        generation.sort(key=lambda x: x[0])
        generation = list(map(lambda x: x[1], generation))

        # show a marker
        print("Generation: {0} -- best Fitness of: {1}".format(i, bestFitness))

        # test if has found member
        if bestFitness == 0:
            return generation[0]

        newGen = list(map(lambda x: pointInversion(x) if random.random() > 1 - 0.7 else x,
                          newGen))  # point mutate with a 70% chance
        generation = generation[:1] + newGen

    # if reached end of iterations and hasn't found perfect member
    return generation[0]

def evolveByRandomSexualReproduction(initialPopulation, fitnessFunction, maxIterations=-1): # Sponges!
    """Evolves a population by randomly mating its members"""
    population = initialPopulation
    if maxIterations == -1:
        maxIterations = sys.maxsize

    for i in range(maxIterations):
        generation = evaluateGeneration(population,
                                        fitnessFunction)  # FIXME: do not hardcode numbers (except perhaps referencing them from somewhere else)
        bestFitness = min(map(lambda x: x[0], generation))
        generation.sort(key=lambda x: x[0])
        generation = list(map(lambda x: x[1], generation))

        # show a marker
        print("Generation: {0} -- best Fitness of: {1}".format(i, bestFitness))

        # test if has found member
        if bestFitness == 0:
            return generation[0]

        # mate with adjacent member
        newGen = [crossover(generation[i], generation[i + 1]) for i in range(len(generation) - 1)]
        newGen = list(map(lambda x: pointInversion(x) if random.random() > 1 - 0.3 else x,
                          newGen))  # point mutate with a 30% chance
        generation = generation[:1] + newGen

    # if reached end of iterations and hasn't found perfect member
    return generation[0]
    
def evolveByRankedSexualReproduction(initialPopulation, fitnessFunction, maxIterations=-1): # Monkeys!
    """Evolves a population by mating members with adjacent fitness"""
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
        
        # mate with random member

        newGen = [crossover(random.choice(generation), random.choice(generation)) for i in range(len(generation)-1)]
        newGen = list(map(lambda x: pointInversion(x) if random.random() > 1-0.3 else x, newGen)) # point mutate with a 30% chance
        generation = generation[:1] + newGen
    
    # if reached end of iterations and hasn't found perfect member
    return generation[0]

def evolveByRankedSexualReproductionWithCarryOverAndRandomMutations(initialPopulation, fitnessFunction, maxIterations=-1): # Birds!
    """Evolves a population by mating members with a similar fitness and keeping the best members from the previous generation"""
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
# BASIC SEARCH FUNCTIONS
################################################################################

# Based on the symmetric heuristic, we target the "max graph" in order to reduce that so that the max graph and the
# min graph are roughly equivalent.


def searchAndDestroy(graph, cliqueSize, fitness):
    """Selects a random clique of a size, toggles edges from that, and then checks the fitness. Returns None if there
    was no improvement"""

    graph.getMax()
    best_fitness = fitness(graph, cliqueSize)
    list_of_cliques = graph.findCliques(cliqueSize)

    def test_graph(clique):
        copygraph = graph.deepcopy()
        copygraph.toggleClique(clique)
        return copygraph

    results = sorted(pool.map(test_graph, list_of_cliques), key=lambda x: fitness(x, cliqueSize))
    if len(results) != 0 and fitness(results[0], cliqueSize) < best_fitness:
        return results[0]

    return None

def bruteForce(graph, cliqueSize, fitness):
    """Toggles all edges, checks which one gave the lowest fitness, returns that"""
    graph.getMax()
    best_fitness = fitness(graph, cliqueSize)
    best_graph = graph.deepcopy()

    def test_graph(edge):
        copygraph = graph.deepcopy()
        copygraph.toggleEdge(edge[0], edge[1])
        return copygraph

    results = sorted(pool.map(test_graph, list(combinations(range(0, graph.nodes), 2))), key=lambda x: fitness(
        x, cliqueSize))

    if len(results) != 0 and fitness(results[0], cliqueSize) < best_fitness:
        return results[0]
    return best_graph


################################################################################
# BEES
################################################################################

def workerBee(tgraph, cliqueSize, numOfBees, fitness):
    """Takes the best graph, toggles edges randomly, sad and brute forces, finds the best fitness among them"""
    dic = {}
    graph = tgraph.deepcopy()
    graph.toggleRandomEdge()
    for i in range(numOfBees):
        is_true = True

        # Search and destroys until it can't go anymore
        while is_true:
            graphc = searchAndDestroy(graph, cliqueSize, fitness)
            if graphc == None:
                is_true = False
            elif fitness(graphc, cliqueSize) == fitness(graph, cliqueSize):
                is_true = False
            else:
                graph = graphc
            try:
                fitness(graphc, cliqueSize)
            except:
                is_true = False

        # Brute forces until it can't go anymore
        is_true = True
        while is_true:
            graphc = bruteForce(graph, cliqueSize, fitness)
            if fitness(graphc, cliqueSize) == fitness(graph, cliqueSize):
                is_true = False
            else:
                graph = graphc

        dic[graph] = fitness(graph, cliqueSize)
        yield min(dic, key=dic.get)

def scoutBee(graphSize, cliqueSize, numOfBees, fitness):
    """Finds a new graph, and brings its fitness down as low as it can go using bruteforce and search and destroy."""
    dic = {}
    for i in range(numOfBees):
        graph = Graph(randomGenerator, graphSize)

        is_true = True
        #Search and destroys until it can't go anymore

        while is_true:
            graphc = searchAndDestroy(graph, cliqueSize, fitness)
            if graphc == None:
                is_true = False
            elif fitness(graphc, cliqueSize) == fitness(graph, cliqueSize):
                is_true = False
            else:
                graph = graphc
            try:
                fitness(graphc, cliqueSize)
            except:
                is_true = False

        #Brute forces until it can't go anymore

        is_true = True
        while is_true:
            graphc = bruteForce(graph, cliqueSize, fitness)
            if fitness(graphc, cliqueSize) == fitness(graph, cliqueSize):
                is_true = False
            else:
                graph = graphc
        dic[graph] = fitness(graph, cliqueSize)
        yield min(dic, key=dic.get)

def lazyBee(graph, cliqueSize, numOfBees, fitness):
    """Takes the best graph, toggles an edge, then runs bruteforce until it can't go any further. Returns the same graph 
    if it doesn't find a better edge."""
    bestGraph = graph
    bestFitness = fitness(graph, cliqueSize)
    isTrue = True
    while isTrue:
        isTrue = False
        for i in range(numOfBees):
            graphc = graph.deepcopy()
            graphc.toggleRandomEdge()
            graphc = searchAndDestroy(graphc, cliqueSize, fitness)
            if graphc == None:
                pass
            elif fitness(graphc, cliqueSize) < bestFitness:
                isTrue = True
                bestGraph = graphc
                bestFitness = fitness(graphc, cliqueSize)
    return bestGraph

def beeMethod(populationSize, numberOfRuns, cliqueSize, graph, fitness):
    """ Generally runs through the bee method using a fitness function. For each iteration, 
    runs through the worker bee, scout bee, and lazy bee, finding the best fitness among them."""
    import math

    # Initialize the bestGraph and bestErr variables.
    best_graph = graph
    best_err = fitness(best_graph, cliqueSize)
    sizeOfGraph = graph.nodes
    print('Start error: ' + str(best_err))

    # Check to see if the fitness is 0
    if best_err == 0:
        return best_graph

    # Trying to find the sweet spot for the population
    numWorker = math.floor(populationSize * 0.5)
    numScout = math.floor(populationSize * 0.1)
    numLazy = populationSize - numScout - numWorker

    print("Number of worker bees: " + str(numWorker))
    print("Number of scout bees: " + str(numScout))
    print("Number of lazy bees: " + str(numLazy))

    # Main loop
    for i in range(numberOfRuns):
        print("Iteration: " + str(i))

        print('Worker bees')
        # Run worker bees
        for i in workerBee(best_graph, cliqueSize, numWorker, fitness):
            if fitness(i, cliqueSize) < best_err:
                best_graph = i
                best_err = fitness(i, cliqueSize)
            if fitness(i, cliqueSize) == 0:
                break

        # Check to see if the fitness is 0
        if best_err == 0:
            print('Found winner')
            return best_graph

        print('Best error so far: ' + str(best_err))

        print('Scout bees')
        for i in workerBee(Graph(randomGenerator, sizeOfGraph), cliqueSize, numScout, fitness):
            if fitness(i, cliqueSize) < best_err:
                best_graph = i
                best_err = fitness(i, cliqueSize)
                break

        # Check to see if the fitness is 0
        if best_err == 0:
            print('Found winner')
            return best_graph


        print('Best error so far: '+str(best_err))
        print('Lazy bees')

        # Run lazy bees

        new_graph = lazyBee(best_graph, cliqueSize, numLazy, fitness)
        if fitness(new_graph, cliqueSize) < best_err:
            best_graph = new_graph
            best_err = fitness(new_graph, cliqueSize)

        # Check to see if the fitness is 0
        if best_err == 0:
            print('Found winner')
            return best_graph

        print('Best error so far: ' + str(best_err))

    print('Best error: ' + str(best_err))
    return best_graph

def buildUpBees(populationSize, graphPop, numberOfRuns, cliqueSize, startSize, endSize, fitness):
    import time
    start = time.time()
    """Starting from the given start size, keep progressively building Ramsey graphs until you reach the endSize"""
    size = startSize
    print("Current size: " + str(size))
    print("Generating population...")
    pop = [Graph(randomGenerator, size) for i in range(graphPop)]
    pop.sort(key=lambda x: fitness(x, cliqueSize))
    best_graph = beeMethod(populationSize, numberOfRuns, cliqueSize, pop[0], fitness)
    size +=1
    while size <= endSize:
        print('Current size: ' + str(size))
        print('Generating population...')
        # The 50 is currently temporary.
        pop = generatePopulation(best_graph, size, graphPop)
        pop.sort(key = lambda x: fitness(x, cliqueSize))
        for i in pop[:20]:
            start = time.time()
            best_graph = beeMethod(populationSize, numberOfRuns, cliqueSize, i, fitness)
            if fitness(best_graph, cliqueSize) == 0:
                break
        if fitness(best_graph, cliqueSize) != 0:
            print('Failed on ' + str(size))
            return best_graph
        size +=1
        print("Elapsed time:")
        print(time.time() - start)

    return best_graph


################################################################################
# SIMULATED ANNEALING
################################################################################

""" The following parameters are considered:"""
""" STATESPACE: All Graphs of a given size"""
""" ENERGY (FITNESS) GOAL: Obviously 0 """
""" THE CANDIDATE GENERATOR PROCEDURE: neighbor() takes a random graph and breaks it down as far as it goes"""
""" ACCEPTANCE PROBABILITY FUNCTION: assess_probability() takes the graph, assess it compared to the best graph, and
    moves accordingly."""
""" ANNEALING SCHEDULE: temperature() generates the temperature according to the graph and time."""
""" INIT TEMP: temp"""

import math

def generateTestGraph(graph, cliqueSize, fitness):
    """Finds a neighbor graph."""
    graph.toggleRandomEdge()

    is_true = True
    # Search and destroys until it can't go anymore

    while is_true:
        graphc = searchAndDestroy(graph, cliqueSize, fitness)
        if graphc == None:
            is_true = False
        elif fitness(graphc, cliqueSize) == fitness(graph, cliqueSize):
            is_true = False
        else:
            graph = graphc
        try:
            fitness(graphc, cliqueSize)
        except:
            is_true = False

    # Brute forces until it can't go anymore

    is_true = True
    while is_true:
        graphc = bruteForce(graph, cliqueSize, fitness)
        if fitness(graphc, cliqueSize) == fitness(graph, cliqueSize):
            is_true = False
        else:
            graph = graphc
    return graph

def neighbor(graph, cliqueSize, fitness):
    """Tries to find the best neighbor graph out of a selection of neighbor graphs"""
    x = [generateTestGraph(graph, cliqueSize, fitness) for i in range(3)]
    x.sort(key=lambda x: fitness(x, cliqueSize))
    return x[0]

# BASIC ASSESS PROBABILITY FUNCTION, IMPROVE OVER TIME.
def assessProbability(graph, temperature, best_err, cliqueSize, fitness):
    """Assesses the probability based on an exponential distribution."""
    return math.exp((best_err - fitness(graph, cliqueSize))/temperature)

def simulatedAnnealing(start_temp, size, cliqueSize, fitness):
    """Finds the global minimum using simulated annealing. Attempts to lower randomness by selecting optimal neighbors."""
    import random
    current_graph = Graph(randomGenerator, size)
    current_err = fitness(current_graph, cliqueSize)
    for i in range(start_temp):
        temp = start_temp - i
        print('Temperature: ' + str(temp) + " - Best error so far: " + str(current_err))
        new_graph = neighbor(current_graph, cliqueSize, fitness)
        if fitness(new_graph, cliqueSize) == 0:
            print('Found winner')
            return new_graph
        elif assessProbability(new_graph, temp, current_err, cliqueSize, fitness) > random.random():
            current_graph = new_graph
            current_err = fitness(new_graph, cliqueSize)
    print(current_err)
    return current_graph

def simulatedAnnealingRandom(start_temp, size, cliqueSize, fitness):
    """Finds the global minimum using simulated annealing. Completely random."""
    import random
    current_graph = Graph(randomGenerator, size)
    current_err = fitness(current_graph, cliqueSize)
    for i in range(start_temp):
        temp = start_temp - i
        print('Temperature: ' + str(temp) + " - Best error so far: " + str(current_err))
        new_graph = generateTestGraph(current_graph, cliqueSize, fitness)
        if fitness(new_graph, cliqueSize) == 0:
            print('Found winner')
            return new_graph
        elif assessProbability(new_graph, temp, current_err, cliqueSize, fitness) > random.random():
            current_graph = new_graph
            current_err = fitness(new_graph, cliqueSize)
    print(current_err)
    return current_graph