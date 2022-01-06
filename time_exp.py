import numpy as np
import graph_tool.all as gt
import scipy.io
import time
import modules.distance_matrix as distance_matrix
from modHMDS import HMDS
from MDS_classic import MDS
import modules.graph_io as graph_io


#G = graph_io.load_graph("graphs/dwt_419.vna")
G = gt.lattice([5,5])
d = distance_matrix.get_distance_matrix(G,'spdm',normalize=True)

def stochastic(d):
    Y = HMDS(d)
    Y.solve()

def classic(d):
    Y = MDS(d,geometry='hyperbolic')
    Y.solve()


def timing(f, n, a):
    print(f.__name__)
    r = range(n)
    t1 = time.perf_counter()
    for i in r:
        f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a)
    t2 = time.perf_counter()
    print((t2-t1)/(10*n))

timing(stochastic,1,d)
timing(classic,1,d)
