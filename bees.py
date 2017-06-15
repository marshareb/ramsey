from graph import *
import random

#Based on the symmetric heuristic, we target the "max graph" in order to reduce that so that the max graph and the min graph
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


def workerBee(graph, cliqueSize, numOfBees):
    """Takes the best graph, toggles edges randomly, finds the best fitness among them"""
    dic = {}
    for i in range(numOfBees):
        a = graph.deepcopy()
        a.toggleRandomEdge()
        dic[a] = a.fitness(cliqueSize)
    return min(dic, key=dic.get)

def scoutBee(graphSize, cliqueSize):
    """Finds a new graph, and brings it down to as low as it can go"""
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
    return graph

def lazyBee(graph, cliqueSize, numOfBees):
    """Takes the best graph, toggles an edge, then runs bruteforce until it can't go any further. Returns None if it 
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