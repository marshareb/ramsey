from itertools import combinations
# from functools import reduce
import random
import networkx as nx
import matplotlib.pyplot as plt
from extensions import flatten, triangleReduction

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

        """
        Networkx clique functions: Used to double check our clique function
        def length_of_lexicon(g, size):
            length = sum(1 for x in lexicon_of_size(g, size))
            return length

        def lexicon_of_size(G, size):
            from collections import deque
            from itertools import chain, islice
            index = {}
            nbrs = {}
            for u in G:
                index[u] = len(index)
                # Neighbors of u that appear after u in the iteration order of G.
                nbrs[u] = {v for v in G[u] if v not in index}

            queue = deque(([u], sorted(nbrs[u], key=index.__getitem__)) for u in G)

            # Loop invariants:
            # 1. len(base) is nondecreasing.
            # 2. (base + cnbrs) is sorted with respect to the iteration order of G.
            # 3. cnbrs is a set of common neighbors of nodes in base.
            while queue:
                base, cnbrs = map(list, queue.popleft())
                if len(base) == size:
                    yield base
                for i, u in enumerate(cnbrs):
                    # Use generators to reduce memory consumption.
                    queue.append((chain(base, [u]),
                                  filter(nbrs[u].__contains__,
                                         islice(cnbrs, i + 1, None))))
        print(length_of_lexicon(A, 4) + length_of_lexicon(B, 4))
        """
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

    def getMax(self):
        """Changes the graph to be the maximum between it and it's complement"""
        if len(self.edgeList()) < len(self.complement().edgeList()):
            self.graph = self.complement().graph

    def generator(self, r, c):
        """a generator which builds a copy and then, if the number of nodes is bigger, fills in the difference 
        randomly"""
        try:
            return self.graph[r][c]
        except:
            return random.choice([True, False])

    def deepcopy(self):
        """Returns a copy of the graph"""
        return Graph(self.generator, self.nodes)

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
        n = [self.hasEdge(node, i) for i in range(self.nodes)]
        return list(map(lambda ib: ib[0], filter(lambda ib: ib[1], enumerate(n))))

    def hasEdge(self, fromNode, toNode):
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

    def toggleRandomEdge(self):
        x = self.edgeList()
        if len(x) == 0:
            x = self.complement().edgeList()
        x = random.choice(x)
        self.toggleEdge(x[0], x[1])


    def toggleEdge(self, row, col):
        # Switch the edges, since we need the larger one to be out front
        if row > col:
            self.graph[row-1][col] = not self.graph[row-1][col]
        else:
            self.graph[col-1][row] = not self.graph[col-1][row]

    ################################################################################

    def findCliques(self, cliqueSize):
        # Generate a list of all clique possibilities
        ls = list(combinations(range(0,self.nodes), cliqueSize)) #create all combinations (order doesn't matter)
        cs = list(map(lambda c: (c, all(self.hasEdge(c[i], c[j]) for i in range(len(c)) for j in range(len(c)) if i!= j)), ls))
        ds = list(map(lambda c: (c, all(not self.hasEdge(c[i], c[j]) for i in range(len(c)) for j in range(len(c)) if i != j)), ls))
        cs = list(filter(lambda b: b[1], cs))
        ds = list(filter(lambda b: b[1], ds))
        cs = list(map(lambda c: c[0], cs))
        ds = list(map(lambda c: c[0], ds))
        # cs here denotes the current graph cliques, ds denotes the anti-cliques or the independent sets.
        return (cs, ds)

    def fitness(self, cliqueSize):
        """returns all cliques and anti-cliques of a given size found in the graph"""
        cliques = self.findCliques(cliqueSize)
        return len(cliques[0]) + len(cliques[1])

    def dna(self):  # get a graph's DNA
        return flatten(self.graph)

    def fromDna(dna):  # birth a graph from DNA
        t = triangleReduction(len(dna))
        if t:
            return Graph(dnaGenerator(dna), t + 1)
        else:
            raise Exception("wrong DNA length - must be a triangle number")
def randomGenerator(r, c):
    return random.choice([True, False])

def generatePopulation(startGraph, graphSize, populationSize):
    """From a prior Ramsey graph, builds a new Ramsey graph"""
    return [Graph(startGraph.generator, graphSize) for x in range(populationSize)]

def dnaGenerator(dna):
    return lambda r, c: dna[int(r*(r+1)/2+c)]