# Ramsey Algorithms

Here are a collection of algorithms written in Python in order to find the lower bound of diagonal Ramsey numbers. We define a Ramsey number v = R(m, n) to be the minimum number of vertices such that all 
graphs of order at least v will have either a clique of size m or an independent set of size n. A diagonal Ramsey number is is the Ramsey number for R(m,m), which will be abbreviated to R(m). Currently, we have
that the lower bound for R(5) is 43. We would like to use these methods to improve this from 43 to 44.

### Use

First clone the repo, then run

```
pip install -r requirements.txt 
```

Then, to see all of the examples, run 

```
python3 ramsey.py 
```

or 

```
python ramsey.py 
```

Depending on if you have Python 2. Note that this was created in Python 3. If you would like to edit the default settings, you can find them in ramsey.py.


## Examples

In order to generate a random example, run 

a = Graph(randomGenerator, n)

where n is a natural number greater than 0.

If you wanted to try finding a counterexample to R(4,4) at n=17, and you wanted to build up from n=12, you could use

buildUpBees(800, 100, 200, 4, 12, 17)

This gives you a population size of 800 bees, it checks through 100 graphs, and it'll do 200 iterations before ending.


## References

[1] Aija'am, Jihad Mohamad. "Can genetic algorithms with the symmetric heuristic find the Ramsey number R (5, 5)." Informatics and Systems (INFOS), 2010 The 7th International Conference on. IEEE, 2010.

[2] Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008

[3] Hunter, John D. "Matplotlib: A 2D graphics environment." Computing In Science & Engineering 9.3 (2007): 90-95.

[4] Zhang, Yun, et al. "Genome-scale computational approaches to memory-intensive applications in systems biology." Supercomputing, 2005. Proceedings of the ACM/IEEE SC 2005 Conference. IEEE, 2005.


## Acknowledgements

Thanks to William Marrujo for his help and guidance, to Dr. Mark Ward for his inspiration and support, and to Dr. David McReynolds for his guidance. This work is supported by NSF grant DSM #1246818.


