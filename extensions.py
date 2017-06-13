from itertools import permutations
from math import sqrt

def necklaces(things, size):
    """takes a list and returns all cyclic permutations of that list, not including their flipped versions"""
    ps = list(permutations(things, size))
    
    i = 0
    while i < len(ps):
        p = ps[i] # a single permutation
        rs = [shift(p, x) for x in range(size)] # foreward rotations of p
        rs += list(map(lambda x: tuple(reversed(x)), rs)) # backward rotations of p
        
        for r in rs[1:]: # for each rotation (except original)
            ps.remove(r) # remove it from the final list of permutations
        
        i += 1
    
    return ps

def shift(l, n):
    """shifts a list n to the right and brings the last elements around to the front"""
    return l[n:] + l[:n]

def flatten(m):
    """flattens a list of lists to 1 dimension lower with all lists appended one after another"""
    return [item for row in m for item in row]

def triangleReduction(n):
    """returns the corresponding number if the number is a triangle number of that number, False otherwise"""
    # m*(m+1)/2 = n
    m = (sqrt(1+8*n) - 1)/2 
    return int(m) if int(m) == m else False
