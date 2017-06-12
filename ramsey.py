# TODO: Read the summary of http://ieeexplore.ieee.org/document/5461802/. It talks a bit about how we can construct counterexamples from prior counterexamples. Maybe James needs to prove that all Ramsey graphs Gi exist as subgraphs of graphs G(i+1)?
# TODO: In the checking for all cliques, we can take a large subset instead of scanning the entire thing, narrowing down all possibilities to a smaller subset to search through.

# uses itertools to generate all tuples for k-clique checking
from itertools import combinations
# reduce container items
from functools import reduce
# uses random for the random graph generator
import random
# to time functions
import time


################################################################################
# REPRESENTATION
################################################################################

class Graph:
    # Instance Variables

    # Information

    def __len__(self):
        return self.nodes

    def __str__(self):  # called with print() or str()
        # the X's represent nodes which are guaranteed to be false, but are technically not part of the storage of the graph
        return "X\n" + "\n".join([" ".join(["T" if node else "F" for node in row]) + " X" for row in self.graph])

    def __repr__(self):  # called if just stated in top level
        return self.__str__()  # defer to a single display function

    # Initialization
    # perhaps with supplying a generator function?

    def __init__(self, generator, nodes):
        self.nodes = nodes
        self.graph = [[generator(row, col) for col in range(0, row + 1)] for row in
                      range(0, nodes - 1)]  # assuming immutability

    def complement(self):
        """Returns a complement graph"""
        gen = lambda r, c: not self.graph[r][c]
        return Graph(gen, self.nodes)

    # Accessors

    def __getitem__(self, key):
        return self.graph[key - 1]

    def __setitem__(self, key, value):
        self.graph[key - 1] = value

    # Methods

    def toggleEdge(self, row, col):
        self.graph[row][col] = not self.graph[row][col]

    def degreeOfNode(self, node):
        r = self.graph[node - 1]
        c = [row[node] for row in self.graph[node:]]
        return sum(r + c)

    def getNeighbors(self, node):
        """Returns all nodes distance 1 from the node given as a list of node numbers"""
        n = self.graph[node - 1] + [False] + [row[node] for row in self.graph[node:]]
        return list(map(lambda ib: ib[0], filter(lambda ib: ib[1], enumerate(n))))

    def haveEdge(self, fromNode, toNode):
        if fromNode == toNode:
            return False
        elif fromNode < toNode:
            fromNode, toNode = toNode, fromNode  # swap positions

        return self.graph[fromNode - 1][toNode]

    def findCliques(self, cliqueSize):
        cs = list(combinations(range(0, self.nodes), cliqueSize))  # get all combinations of possible cliques
        cs = list(map(lambda c: (c, [(c[i - 1], x) for i, x in enumerate(c)]), cs))  # make pairs of beginning and end points of edges along clique
        cs = list(map(lambda l: (l[0], [self.haveEdge(x[0], x[1]) for x in l[1]]), cs))  # evaluate each of those pairs and see if the edge exists
        cs = list(map(lambda l: (l[0], reduce(lambda a, b: a and b, l[1])), cs))  # see if clique has any nonexistant edges
        cs = list(filter(lambda b: b[1], cs))  # take only the ones that have all existing edges
        cs = list(map(lambda b: b[0], cs))  # get its associated node tuple (the one that it's been passing along this whole time)
        return cs

    # Stuff James has added 6/8/17
    def fitness(self, cliqueSize):
        """Gets the overall fitness (distance from being a Ramsey Graph)"""
        """Alternative fitness function -- add up cliques from a test clique set instead of adding up all of the cliques"""
        return len(self.findCliques(cliqueSize)) + len(self.complement().findCliques(cliqueSize))

    #Randomly select an edge and toggle it
    def mutate(self):
        x = random.randint(0,len(self.graph)-1)
        y = random.randint(0, len(self.graph[x])-1)
        self.toggleEdge(x,y)

    #Uses networkx and matplotlib to plot the graph
    def draw(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.Graph()
        node_list = []
        edge_list = []
        labels = {}

        for i in range(len(self.graph)+1):
            G.add_node(i)
            node_list.append(i)
            labels[i] = i

        pos = nx.circular_layout(G)
        for i in range(len(self.graph)+1):
            for j in range(len(self.graph)+1):
                if self.haveEdge(i,j):
                    G.add_edge(i,j)
                    edge_list.append((i,j))

        nx.draw_networkx_nodes(G, pos, nodelist = node_list)
        nx.draw_networkx_edges(G, pos, edgelist = edge_list)
        nx.draw_networkx_labels(G, pos, labels)
        plt.show()



# TODO: Need to add some way to merge graphs. Merging graphs involves selecting a random number of edges from one graph and a random number of edges from another graph and then putting those edges together
# (Edges in this case includes the possibility of not having an edge. Essentially grabs a random number of boolean values from one graph and a random number of boolean values from another graph and merges them this way.)

# NOTE: We only care about what are called diagonal Ramsey numbers (since R(5,5) is our end goal).
# Let's create a sufficiently random population of graphs. For crossover (or mating) we simply take random values from each graph. If the fitness is better than our worst fitness, we then add this graph to our population
# We should also have an efficient way to output the data to a text file for analysis


################################################################################
# TESTING
################################################################################

def randomGenerator(r, c):
    return random.choice([True, False])


def ramseyTest(populationSize, numberOfRuns, cliqueSize, size):
    import sys
    #Dictionary to hold all of the different graphs associated to their fitness number
    pop = {}

    #Initialize variables
    bestFitness = sys.maxsize
    bestGraph = Graph(0, 0)
    worstFitness = 0

    #Generate all the graphs, figure out bestFitness and worstFitness
    for i in range(populationSize):
        a = Graph(randomGenerator, size)
        pop[a] = a.fitness(cliqueSize)
        if pop[a] < bestFitness:
            bestFitness = pop[a]
            bestGraph = a
        if pop[a] > worstFitness:
            worstFitness = pop[a]

    #Check to see if we've already finished
    if bestFitness == 0:
        return bestGraph

    # Run through the genetic simulation
    for i in range(numberOfRuns):
        # That way, we can keep track of if it's running and how long each iteration is taking
        print('Iteration: ' + str(i))
        # Mutate each graph
        for k in pop:
            k.mutate()
            pop[k] = k.fitness(cliqueSize)
            if pop[k] < bestFitness:
                bestFitness = pop[k]
                bestGraph = k
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

    #Once we've cycled through all of the numberOfRuns, we conclude by saying how close we got to our goal of 0.
    print(bestFitness)
    return bestGraph

################################################################################
# EXECUTION
################################################################################

def testRamsey():
    a = ramseyTest(10, 50, 4, 6)
    print(a.fitness(4))
    print(a.findCliques(4))
    print(a)
    a.draw()

if __name__ == "__main__": # if python script is run as an executable
    testRamsey()
