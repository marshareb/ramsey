from graph import *
import random
import sys
from ramsey import *

# Based on the symmetric heuristic, we target the "max graph" in order to reduce that so that the max graph and the min graph
# are roughly equivalent.
def searchAndDestroy(graph, cliqueSize):
    """Selects a random clique of a size, toggles edges from that, and then checks the fitness. Returns None if there
    was no improvement"""
    """ISSUE: it's very slow right now. Generally, if it doesn't find the clique right away, it won't find it at all."""
    graph.getMax()
    bestFitness = graph.fitness(cliqueSize)
    list_of_cliques = graph.findCliques(cliqueSize)[0]
    while len(list_of_cliques) != 0:
        copygraph = graph.deepcopy()
        clique1 = list(random.choice(list_of_cliques))
        clique = clique1.copy()
        for node1 in clique:
            clique.remove(node1)
            neighbor_list = graph.getNeighbors(node1)
            for i in neighbor_list:
                if i not in clique:
                    neighbor_list.remove(i)
            for i in neighbor_list:
                copygraph.toggleEdge(node1, i)
                if copygraph.fitness(cliqueSize) < bestFitness:
                    return copygraph
        list_of_cliques.remove(tuple(clique1))
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
    numWorker = math.floor(populationSize * 0.56)
    numScout = 10
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
