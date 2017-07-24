from itertools import combinations
import random
import networkx as nx
import matplotlib.pyplot as plt
from extensions import triangleReduction

# DNA BIT FORMAT:
# The format is a boolean bit which builds consecutively starting from the zero node.

# EXAMPLE: Graph of size 3 which has the edge list [(0,1), (1,2)]
# [True, False, True] <-> [(0,1), (0,2), (1,2)]

class Graph:
    # Instance Variables
    
    # Information
    
    def __len__(self):
        """Return the number of nodes in the graph."""
        return self.nodes

    def __str__(self): # called with print() or str()
        """Draws the graph in a matrix format."""
        # the X's represent nodes which are guaranteed to be false, but are technically not part of the storage of the graph
        return "X\n" + "\n".join([" ".join(["T" if node else "F" for node in row]) + " X" for row in self.graph])
    
    def __repr__(self): # called if just stated in top level
        return self.__str__() # defer to a single display function

    # Reference:  [2] Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics,
    # and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux,
    # Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008


    def draw(self):
        """Draws the graph using a Furchterman Reignold layout."""
        
        # Gather graph information
        nodes = list(range(self.nodes))
        labels = {x: x for x in nodes}
        potential_edges = list(combinations(range(self.nodes), 2)) # potential edges
        edges = [(e, self.hasEdge(e[0], e[1])) for e in potential_edges] # associate edge with wether it exists
        edges = list(filter(lambda x: x[1], edges)) # filter out nonexistant edges
        edges = list(map(lambda x: x[0], edges)) # get back associated edge tuple
        antiedges = [e for e in potential_edges if e not in edges] # get complement list

        # Create graph object
        a = nx.Graph() # create nx graph
        a.add_nodes_from(nodes) # add nodes
        a.add_edges_from(edges) # add edges
        
        b = nx.Graph() # create complement graph
        b.add_nodes_from(nodes) # add nodes (same)
        b.add_edges_from(antiedges) # add edges (opposite)
        
        # Draw Graph
        pos_a = nx.fruchterman_reingold_layout(a)
        pos_b = nx.fruchterman_reingold_layout(b)
        
        plt.figure(1)
        
        plt.subplot(1, 2, 1)
        plt.title("Graph")
        nx.draw_networkx_nodes(a, pos_a, nodelist=nodes)
        nx.draw_networkx_edges(a, pos_a, edgelist=edges)
        nx.draw_networkx_labels(a, pos_a, labels)
        
        plt.subplot(1, 2, 2)
        plt.title("Complement")
        nx.draw_networkx_nodes(b, pos_b, nodelist=nodes)
        nx.draw_networkx_edges(b, pos_b, edgelist=antiedges)
        nx.draw_networkx_labels(b, pos_b, labels)
        
        plt.show()

    def draw2(self):
        """Draws the graph using a ciruclar layout."""

        # Gather graph information
        nodes = list(range(self.nodes))
        labels = {x: x for x in nodes}
        potential_edges = list(combinations(range(self.nodes), 2))  # potential edges
        edges = [(e, self.hasEdge(e[0], e[1])) for e in potential_edges]  # associate edge with wether it exists
        edges = list(filter(lambda x: x[1], edges))  # filter out nonexistant edges
        edges = list(map(lambda x: x[0], edges))  # get back associated edge tuple
        antiedges = [e for e in potential_edges if e not in edges]  # get complement list

        a = nx.Graph()
        a.add_nodes_from(nodes)
        a.add_edges_from(edges)

        b = nx.Graph()
        b.add_nodes_from(nodes)
        b.add_edges_from(antiedges)

        pos = nx.circular_layout(a)

        nx.draw_networkx_nodes(a, pos, nodelist=nodes)
        nx.draw_networkx_edges(a, pos, edgelist=edges, edge_color = 'r')
        nx.draw_networkx_labels(a, pos, labels)
        nx.draw_networkx_edges(b, pos, edgelist=antiedges, edge_color = 'b')
        plt.show()

    def writeToFile(self, filename):
        """Writes dna to a file."""
        f = open(filename, 'w')
        for i in self.dna():
            f.write(str(i) + "\n")
        f.close()

    # Initialization

    def __init__(self, generator, nodes):
        """Initializes graph."""
        if nodes < 0:
            raise Exception("Number of nodes needs to be greater than 0.")
        self.nodes = nodes
        self.graph = [[generator(row, col) for col in range(0, row + 1)] for row in range(0, nodes - 1)] # assuming immutability

    def complement(self):
        """Returns a complement graph."""
        gen = lambda r, c: not self.graph[r][c]
        return Graph(gen, self.nodes)
    
    # Accessors

    def getMax(self):
        """Changes the graph to be the graph which has the most edges between it and it's complement."""
        if len(self.edgeList()) < len(self.complement().edgeList()):
            self.graph = self.complement().graph

    def generator(self, r, c):
        """A generator which builds a copy and then, if the number of nodes is bigger, fills in the difference 
        randomly."""
        try:
            return self.graph[r][c]
        except:
            return random.choice([True, False])

    def deepcopy(self):
        """Returns a copy of the graph."""
        return Graph(self.generator, self.nodes)

    def __getitem__(self, key):
        """Get value of the edge."""
        if key == 0:
            return []
        return self.graph[key - 1]
    
    def __setitem__(self, key, value):
        """Set the value of an edge."""
        self.graph[key - 1] = value
    
    # Methods
    
    def degreeOfNode(self, node):
        """Returns the degree of a node."""
        r = self.graph[node - 1]
        c = [row[node] for row in self.graph[node:]]
        return sum(r + c)
    
    def getNeighbors(self, node):
        """Returns all nodes distance 1 from the node given as a list of node numbers."""
        n = [self.hasEdge(node, i) for i in range(self.nodes)]
        return list(map(lambda ib: ib[0], filter(lambda ib: ib[1], enumerate(n))))

    def hasEdge(self, fromNode, toNode):
        """Returns True if there is an edge between the nodes, and False otherwise."""
        if fromNode == toNode:
            return False
        elif fromNode < toNode:
            fromNode, toNode = toNode, fromNode # swap positions
    
        return self.graph[fromNode - 1][toNode]

    def edgeList(self):
        """Returns all the edges in the graph"""
        # The j < i condition ensures that all of them are unique.
        return [(i, j) for i in range(len(self.graph) + 1) for j in range(len(self.graph) + 1) if
                self.hasEdge(i, j) and j < i]

    def toggleRandomEdge(self):
        """Toggles a random edge from the graph."""
        x = self.edgeList()
        if len(x) == 0:
            x = self.complement().edgeList()
        x = random.choice(x)
        self.toggleEdge(x[0], x[1])

    def toggleEdge(self, row, col):
        """Toggles a selected edge from a graph."""
        # Switch the edges, since we need the larger one to be out front
        if row > col:
            self.graph[row-1][col] = not self.graph[row-1][col]
        else:
            self.graph[col-1][row] = not self.graph[col-1][row]

    def toggleClique(self, clique):
        """Toggles a whole clique at once"""
        list(map(lambda x: self.toggleEdge(x[0],x[1]), list(combinations(clique,2))))

    ################################################################################

    #Reference:  [4] Zhang, Yun, et al. "Genome-scale computational approaches to memory-intensive applications in
    # systems biology." Supercomputing, 2005. Proceedings of the ACM/IEEE SC 2005 Conference. IEEE, 2005.
    def findCliques(self, cliqueSize):
        """ Finds all cliques from a graph using a buildup method. """
        nbrs = {}
        for i in range(self.nodes):
            nbrs[i] = set(self.getNeighbors(i))

        prvsCliqueSize = 2
        clqs = list(map(lambda c: list(c), self.edgeList()))

        while prvsCliqueSize < cliqueSize and len(clqs) != 0:
            nclqs = []
            for k in clqs:
                d = list(map(lambda c: nbrs[c], k))
                for i in set.intersection(*map(set,d)):
                    if i not in k:
                        nclqs.append(k +[i])

            #Remove all permutations
            clqs = list(map(lambda c: sorted(c), nclqs))
            prvsCliqueSize += 1
        return list(set(map(lambda c: tuple(c), clqs)))

    def cliqueDifference(self, cliqueSize):
        """Finds the clique difference (the absolute value of the number of cliques in the graph minus the number of 
        cliques in the complement graph)."""
        return abs(len(self.findCliques(cliqueSize)) - len(self.complement().findCliques(cliqueSize)))

    def dna(self): # get a graph's DNA
        """Gets a graphs DNA (a bit of boolean values which represent the nodes/edges of a graph)."""
        return [self.hasEdge(i,j) for i in range(self.nodes) for j in range(self.nodes) if i < j]

def fromDna(dna):
    """Creates a graph from a DNA bit."""
    t = triangleReduction(len(dna))
    if t:
        edges = [(i,j) for i in range(t+1) for j in range(t+1) if i < j ]
        edges = [(edges[i], dna[i]) for i in range(len(dna))]
        edges = list(filter(lambda x : x[1], edges))
        edges = list(map(lambda x: x[0], edges))
        g = Graph(randomGenerator, t+1)
        for i in g.edgeList():
            if i not in edges:
                g.toggleEdge(i[0], i[1])
        for i in edges:
            if i not in g.edgeList():
                g.toggleEdge(i[0], i[1])
        return g
    else:
        raise Exception("Wrong DNA length - must be a triangle number.")

def boolConvert(s):
    """Converts a string boolean value to an actual boolean value."""
    s = s.strip()
    return s == "True"

def readFromFile(filename):
    """Reads a dna from a file."""
    f = open(filename, 'r')
    dna= []
    with open(filename) as f:
        for line in f:
            dna.append(boolConvert(line))
    return fromDna(dna)

def randomGenerator(r, c):
    """Random generator for a graph."""
    return random.choice([True, False])

def symmetricFitness(graph, cliqueSize):
    """Finds the fitness in the symmetric sense (sum of the difference between cliques up to the number plus 
    the number of cliques in the graph and its complement)."""
    return sum([graph.cliqueDifference(i) for i in range(2, cliqueSize)]) + fitness(graph, cliqueSize)

def fitness(graph, cliqueSize):
    """Finds fitness in the classical sense 
    (number of cliques of the graph plus the number of cliques in the complement graph)."""
    return len(graph.findCliques(cliqueSize)) + len(graph.complement().findCliques(cliqueSize))