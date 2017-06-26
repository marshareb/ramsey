from itertools import combinations
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
        plt.show()

    # Initialization

    def write_to_file(self, filename):
        """Writes dna to a file"""
        f = open(filename, 'w')
        f.write(str(self.dna()))
        f.close()

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

    # leaving this in just for the graph bee case, but for all intents and purpose will be using the pointinversion
    # method
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

    def toggleClique(self, clique):
        """Toggles a whole clique at once"""
        list(map(lambda x: self.toggleEdge(x[0],x[1]), list(combinations(clique,2))))

    ################################################################################

    def findCliques(self, cliqueSize):
        from itertools import permutations
        nbrs = {}
        for i in range(self.nodes):
            nbrs[i] = set(self.getNeighbors(i))

        prvsCliqueSize = 2
        clqs = list(map(lambda c: list(c), self.findCliques(2)[0]))
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

    def fitness(self, cliqueSize):
        return len(self.findCliques2(cliqueSize)) + len(self.complement().findCliques2(cliqueSize))

def randomGenerator(r, c):
    return random.choice([True, False])
