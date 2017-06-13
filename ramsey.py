import sys

from graph import *
from evolution import *

# TODO: Read the summary of http://ieeexplore.ieee.org/document/5461802/. It 
# talks a bit about how we can construct counterexamples from prior 
# counterexamples. Maybe James needs to prove that all Ramsey graphs Gi exist as
# subgraphs of graphs G(i+1)?

# TODO: In the checking for all cliques, we can take a large subset instead of 
# scanning the entire thing, narrowing down all possibilities to a smaller 
# subset to search through.

# TODO: Need to add some way to merge graphs. Merging graphs involves selecting 
# a random number of edges from one graph and a random number of edges from 
# another graph and then putting those edges together (Edges in this case 
# includes the possibility of not having an edge. Essentially grabs a random 
# number of boolean values from one graph and a random number of boolean values 
# from another graph and merges them this way.)

# NOTE: We only care about what are called diagonal Ramsey numbers (since R(5,5)
# is our end goal).
# Let's create a sufficiently random population of graphs. For crossover 
# (or mating) we simply take random values from each graph. If the fitness is 
# better than our worst fitness, we then add this graph to our population
# We should also have an efficient way to output the data to a text file for analysis

################################################################################
# TESTING
################################################################################

# add graph extension for this test case
# TODO: adapt to larger format later
class Graph(Graph):
    def mutate(self):
        x = random.randint(0,len(self.graph)-1)
        y = random.randint(0, len(self.graph[x])-1)
        self.toggleEdge(x,y)
    
    def toggleEdge(self, row, col):
        self.graph[row][col] = not self.graph[row][col]

def ramseyTest(populationSize, numberOfRuns, cliqueSize, size):
    import sys
    # Dictionary to hold all of the different graphs associated to their fitness number
    pop = {}

    # Initialize variables
    bestFitness = sys.maxsize
    bestGraph = Graph(0, 0)
    worstFitness = 0

    # Generate all the graphs, figure out bestFitness and worstFitness
    for i in range(populationSize):
        a = Graph(randomGenerator, size)
        pop[a] = a.fitness(cliqueSize)
        if pop[a] < bestFitness:
            bestFitness = pop[a]
            bestGraph = a
        if pop[a] > worstFitness:
            worstFitness = pop[a]

    # Check to see if we've already finished
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

    # Once we've cycled through all of the numberOfRuns, we conclude by saying how close we got to our goal of 0.
    print(bestFitness)
    return bestGraph

################################################################################
# EXECUTION
################################################################################

def testGraph():
    a = Graph(randomGenerator, 6)
    print(a)
    print(a.fitness(3))
    a.draw()

def testDnaGenerator():
    a = Graph(dnaGenerator([True, True, True, True, True, True]), 4)
    print(a)
    print(a.fitness(3))
    a.draw()

def testRamsey():
    a = ramseyTest(10, 50, 4, 6)
    print(a.fitness(4))
    print(a.findCliques(4))
    print(a)
    a.draw()

if __name__ == "__main__": # if python script is run as an executable
    testRamsey()


    
        
        
        
