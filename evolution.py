import random

from extensions import flatten
from graph import *

################################################################################
# GRAPH EXTENSION
################################################################################

class Graph(Graph):
    def dna(self):
        return flatten(self.graph)

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
    print(crossovers, "crossovers")
    crossoverPoints = random.choices(list(range(0, len(mom))), k=crossovers)
    print(crossoverPoints, "points")
    
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
