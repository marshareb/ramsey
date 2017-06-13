from itertools import combinations, permutations

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
    return l[n:] + l[:n]
