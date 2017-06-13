from itertools import combinations
from functools import reduce
import random
import networkx as nx
import matplotlib.pyplot as plt

from extensions import necklaces

class Graph:
    # Instance Variables
    
    # Information
    
    def __len__(self):
        return self.nodes
    
    def __str__(self): # called with print() or str()
        """draws the graph in a matrix format"""
        # the X's represent nodes which are guaranteed to be false, but are technically not part of the storage of the graph
        return "X\n" + "\n".join([" ".join(["T" if node else "F" for node in row]) + " X" for row in self.graph])
    
    def __repr__(self): # called if just stated in top level
        return self.__str__() # defer to a single display function
    
    def draw(self):
        """draws the node in a visual format"""
        
        # Gather graph information
        nodes = list(range(self.nodes))
        labels = {x: x for x in nodes}
        potentialEdges = list(combinations(range(self.nodes), 2)) # potential edges
        edges = [(e, self.hasEdge(e[0], e[1])) for e in potentialEdges] # associate edge with wether it exists
        edges = list(filter(lambda x: x[1], edges)) # filter out nonexistant edges
        edges = list(map(lambda x: x[0], edges)) # get back associated edge tuple
        antiedges = [e for e in potentialEdges if e not in edges] # get complement list
        
        # Create graph object
        A = nx.Graph() # create nx graph
        A.add_nodes_from(nodes) # add nodes
        A.add_edges_from(edges) # add edges
        
        B = nx.Graph() # create complement graph
        B.add_nodes_from(nodes) # add nodes (same)
        B.add_edges_from(antiedges) # add edges (opposite)
        
        # Draw Graph
        posA = nx.fruchterman_reingold_layout(A)
        posB = nx.fruchterman_reingold_layout(B)
        
        plt.figure(1)
        
        plt.subplot(1, 2, 1)
        plt.title("Graph")
        nx.draw_networkx_nodes(A, posA, nodelist=nodes)
        nx.draw_networkx_edges(A, posA, edgelist=edges)
        nx.draw_networkx_labels(A, posA, labels)
        
        plt.subplot(1, 2, 2)
        plt.title("Complement")
        nx.draw_networkx_nodes(B, posB, nodelist=nodes)
        nx.draw_networkx_edges(B, posB, edgelist=antiedges)
        nx.draw_networkx_labels(B, posB, labels)
        
        plt.show()

    def draw2(self):
        """Draws both colorings on the graph"""

        # Gather graph information
        nodes = list(range(self.nodes))
        labels = {x: x for x in nodes}
        potentialEdges = list(combinations(range(self.nodes), 2))  # potential edges
        edges = [(e, self.hasEdge(e[0], e[1])) for e in potentialEdges]  # associate edge with wether it exists
        edges = list(filter(lambda x: x[1], edges))  # filter out nonexistant edges
        edges = list(map(lambda x: x[0], edges))  # get back associated edge tuple
        antiedges = [e for e in potentialEdges if e not in edges]  # get complement list

        A = nx.Graph()
        A.add_nodes_from(nodes)
        A.add_edges_from(edges)

        B = nx.Graph()
        B.add_nodes_from(nodes)
        B.add_edges_from(antiedges)

        pos = nx.circular_layout(A)

        nx.draw_networkx_nodes(A, pos, nodelist=nodes)
        nx.draw_networkx_edges(A, pos, edgelist=edges, edge_color = 'r')
        nx.draw_networkx_labels(A, pos, labels)
        nx.draw_networkx_edges(B, pos, edgelist=antiedges, edge_color = 'b')

        plt.show()

    # Initialization
    
    def __init__(self, generator, nodes):
        self.nodes = nodes
        self.graph = [[generator(row, col) for col in range(0, row + 1)] for row in range(0, nodes - 1)] # assuming immutability
    
    def complement(self):
        """Returns a complement graph"""
        gen = lambda r, c: not self.graph[r][c]
        return Graph(gen, self.nodes)
    
    # Accessors
    
    def generator(self):
        """returns a generator that generates a copy"""
        return lambda r, c: self.graph[r][c]

    def generator2(self, r, c):
        """a generator which builds a copy and then, if the number of nodes is bigger, fills in the difference 
        randomly"""
        try:
            return self.graph[r][c]
        except:
            return random.choice([True, False])

    def __getitem__(self, key):
        if key == 0:
            return []
        return self.graph[key - 1]
    
    def __setitem__(self, key, value):
        self.graph[key - 1] = value
    
    # Methods
    
    def degreeOfNode(self, node):
        r = self.graph[node - 1]
        c = [row[node] for row in self.graph[node:]]
        return sum(r + c)
    
    def getNeighbors(self, node):
        """Returns all nodes distance 1 from the node given as a list of node numbers"""
        n = self.graph[node - 1] + [False] + [row[node] for row in self.graph[node:]]
        return list(map(lambda ib: ib[0], filter(lambda ib: ib[1], enumerate(n))))
    
    def hasEdge(self, fromNode, toNode):
        if fromNode == toNode:
            return False
        elif fromNode < toNode:
            fromNode, toNode = toNode, fromNode # swap positions
    
        return self.graph[fromNode - 1][toNode]

    ################################################################################
    # JAMES ADDED THIS

    def edgeList(self):
        """Returns all the edges in the graph"""
        # The j < i condition ensures that all of them are unique.
        return [(i, j) for i in range(len(self.graph) + 1) for j in range(len(self.graph) + 1) if
                self.hasEdge(i, j) and j < i]

    def mutate(self, cliqueSize):
        #TODO: fix this so it ramseyTest or any of the other genetic algorithms actually return counterexamples
        # for R(4,4) on 6 vertices.
        import copy
        fit = self.fitness(cliqueSize)
        count = 0

        while self.fitness(cliqueSize) != 0:
            isTrue = True
            for i in self.edgeList():
                a = copy.deepcopy(self)
                a.toggleEdge(i[0]-1,i[1])
                if a.fitness(cliqueSize) < self.fitness(cliqueSize):
                    self.graph = a.graph
                    isTrue = False
                    count +=1
                    return None
            if isTrue == True:
                break
        print(count)
        if self.fitness(cliqueSize) == fit:
            print('No valid mutation, randomly toggling edge')
            x = random.randint(0, len(self.graph) - 1)
            y = random.randint(0, len(self.graph[x]) - 1)
            self.toggleEdge(x, y)


    def toggleEdge(self, row, col):
        self.graph[row][col] = not self.graph[row][col]

    ################################################################################

    def findCliques(self, cliqueSize):
        """returns a tuple of the list of cliques and the list of anti-cliques"""
        cs = list(necklaces(range(0, self.nodes), cliqueSize)) # get all combinations of possible cliques (order matters)
        cs = list(map(lambda c: (c, [(c[i - 1], x) for i, x in enumerate(c)]), cs)) # make pairs of beginning and end points of edges along clique
        cs = list(map(lambda l: (l[0], [self.hasEdge(x[0], x[1]) for x in l[1]]), cs)) # evaluate each of those pairs and see if the edge exists
        cs = list(map(lambda l: (l[0], all(l[1]), not(any(l[1]))), cs)) # record if the clique is all edges or all non-edges
        ds = list(filter(lambda b: b[1], cs)) # take only the ones that have all existing edges (cliques)
        qs = list(filter(lambda b: b[2], cs)) # take only the ones that have all non-existing edges (anti-cliques)
        ds = list(map(lambda b: b[0], ds)) # get its associated node tuple (the one that it's been passing along this whole time)
        qs = list(map(lambda b: b[0], qs)) # get its associated node tuple (the one that it's been passing along this whole time)
        return (ds, qs)
    
    def fitness(self, cliqueSize):
        """returns all cliques and anti-cliques of a given size found in the graph"""
        # cs = list(necklaces(range(0, self.nodes), cliqueSize)) # get all combinations of possible cliques (order matters)
        # cs = list(map(lambda c: (c, [(c[i - 1], x) for i, x in enumerate(c)]), cs)) # make pairs of beginning and end points of edges along clique
        # cs = list(map(lambda l: (l[0], [self.hasEdge(x[0], x[1]) for x in l[1]]), cs)) # evaluate each of those pairs and see if the edge exists
        # cs = list(map(lambda l: (l[0], all(l[1]), not(any(l[1]))), cs)) # record if the clique is all edges or all non-edges
        # cs = list(filter(lambda b: b[1] or b[2], cs)) # take only the ones that have all existing or non-existing edges
        # cs = list(map(lambda b: b[0], cs)) # get its associated node tuple (the one that it's been passing along this whole time)
        # return len(cs)
        cliques = self.findCliques(cliqueSize)
        return len(cliques[0]) + len(cliques[1])

def randomGenerator(r, c):
    return random.choice([True, False])