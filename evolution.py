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
        return flatten(self.graph)
    
    def fromDna(dna): # birth a graph from DNA
        t = triangleReduction(len(dna))
        if t:
            return Graph(dnaGenerator(dna), t+1)
        else:
            raise Exception("wrong DNA length - must be a triangle number")

def dnaGenerator(dna):
    return lambda r, c: dna[int(r*(r+1)/2+c)]

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