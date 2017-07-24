# Ramsey Algorithms

Here are a collection of algorithms written in Python in order to find the lower bound of diagonal Ramsey numbers. We
define a Ramsey number v = R(m, n) to be the minimum number of vertices such that all graphs of order at least v will
have either a clique of size m or an independent set of size n. A diagonal Ramsey number is is the Ramsey number for
R(m,m), which will be abbreviated to R(m). Currently, we have that the lower bound for R(5) is 43. We would like to use
these methods to improve this from 43 to 44.

## Terminology

**Clique:** We say that we have a k-clique if there is a subset of nodes of size k such that each node is adjacent to each
other node in the clique.

**Fitness:** A measure of distance from our goal, which is to have 0 k-cliques.

**Standard Fitness:** It measures the distance from our goal by checking the difference between the number of k-cliques of
the graph and it's complement.

**Symmetric Fitness:** It measure the distance from our goal by ensuring that the graph and its complement are symmetric for
all cliques up to k, and then measures the difference of the k-cliques between the graph and its complement.

**Mutation:** A toggle of some attribute on the graph. In our case, it generally has to do with some collection of edges.

**Standard Mutation:** Toggles a single edge on a graph.

**Clique Mutation:** Toggles an entire clique on a graph.

**DNA:** A boolean bit which models the edges of a graph. The format for the DNA bit in our program is it starts from (0,0)
and goes through all combinations of nodes, where it is of the form (i,j) where i < j, and there are no repeated edges.
For example, for a graph of size 3 which has the edge list [(0,1), (1,2)], we have

[True, False, True] <-> [(0,1), (0,2), (1,2)]

## Installation and Examples

First clone the repo, then run

```
pip install -r requirements.txt 
```

Then, to see all of the preconstructed examples, run

```
python3 ramsey.py 
```

if you have Python 2, or

```
python ramsey.py 
```

if you only have Python 3. Note that this was created in Python 3. If you would like to edit the default
settings, you can find them in ramsey.py. If you would like to just use the algorithms, then you would just run

```
import ramseyAlgorithms
```


## Usage

First, run

```
import ramseyAlgorithms as ra
```

to import the proper package. In order to generate a random graph, run

```
a = ra.Graph(randomGenerator, n)
```

where n is a natural number greater than 0. If you would like to see the standard fitness of this graph, you would run

```
ra.fitness(a, n)
```

where n is the size of the cliques that you want to examine. If you would want to see the symmetric fitness, you would
run

```
ra.symmetricFitness(a,n)
```

where once again, n is the size of the cliques you would want to examine. If you wanted to randomly toggle an edge on
the graph, you would use

```
a.toggleRandomEdge()
```

To see which nodes are the cliques in the graph, you would run

```
a.findCliques(n)
```

where n is as prior.

If you wanted to try finding a counterexample to R(4,4) at n=17, and you wanted to build up from n=12, you could use

```
ra.buildUpBees(800, 100, 200, 4, 12, 17)
```

This gives you a population size of 800 bees, it checks through 100 graphs, and it'll do 200 iterations before ending.

If you wanted to save a graph, you would run

```
a.writeToFile('filename.txt')
```

and if you wanted to then read that file you would run

```
ra.readFromFile('filename.txt')
```

where filename.txt is whatever you wish to name your graph.

## References

[1] Aija'am, Jihad Mohamad. "Can genetic algorithms with the symmetric heuristic find the Ramsey number R (5, 5)."
Informatics and Systems (INFOS), 2010 The 7th International Conference on. IEEE, 2010.

[2] Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using
NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod
Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008

[3] Hunter, John D. "Matplotlib: A 2D graphics environment." Computing In Science & Engineering 9.3 (2007): 90-95.

[4] Zhang, Yun, et al. "Genome-scale computational approaches to memory-intensive applications in systems biology."
Supercomputing, 2005. Proceedings of the ACM/IEEE SC 2005 Conference. IEEE, 2005.


## Acknowledgements

Thanks to William Marrujo for his help and guidance, to Dr. Mark Ward for his inspiration and support, and to Dr. David
McReynolds for his guidance. This work is supported by NSF grant DSM #1246818.
