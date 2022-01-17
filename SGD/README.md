**Hyperbolic Multidimensional Scaling by Stochastic Gradient Descent**

*HMDS.py*   
Contains a class and helper functions to perform HMDS. The interface can be used by:
```
M = HMDS(dissimilarity_matrix)
hyperbolic_embedding = M.solve()
```
The embedding is returned (also stored as M.X) as a numpy array of shape (n,2) where the i'th element of axis 0 are the x and y elements of the i'th datapoint in lobachevsky coordinates.

`HMDS(dissimilarities, init_pos = np.empty(1) )`  
  dissimilarities: Pairwise distance matrix of data  
  init_pos (optional): Starting embedding for the algorithm. If default, positions are assigned at random.

`HMDS().solve(num_iter=20, debug=False, opt_alpha=True)`  
  num_iter: int  
  Maximum number of iterations to perform.

  debug: bool  
  debug=True will save the stress values of the algorithm in self.stress_hist. This can significantly increase the runtime of the algorithm.

  opt_alpha: bool  
  Whether or not to optimize over the scaling parameter. False(default) will use a heuristic, and true will optimize using scipy scalar optimizer.



*graph_processing.py*  
Contains preprocessing functions for graphs.

`get_distance_matrix(graphfile, fmt='dot', lib='gt')`
Returns a numpy matrix with shape (|V|,|V|) where entry i,j is the shortest path between nodes i and j.

graphfile: str or file-like object   
File to load graph data from. If given as a string, will attempt to read the file with that relative path.

fmt: str  
File format of graphfile. Assume to be graphviz .dot file, but graph-tool can support others as described in graph-tool documentation

lib: str
Graph library to work with. If graph-tool is not possible or too difficult to setup, can pass 'nx' to use the networkx python library instead. Preprocessing may take much longer for large graphs.


*MDS_classic*  
Contains an implmentation of MDS via gradient descent. Interface is the same as HMDS.py.


*SGD_MDS*   
Contains an implmentation of graph drawing by stochastic gradient descent, described here: https://github.com/jxz12/s_gd2. Interface is again the same as HMDS.py.
