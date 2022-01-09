import numpy as np
import graph_tool.all as gt
import scipy.io
import time
import modules.distance_matrix as distance_matrix
from modHMDS import HMDS
from MDS_classic import MDS,solve2
import modules.graph_io as graph_io
import networkx as nx

import itertools


def stochastic(d):
    Y = HMDS(d)
    Y.solve()


def classic(d):
    Y = MDS(d,geometry='hyperbolic')
    solve2(Y.X,Y.d,Y.w,debug=False)


def timing(f, n, a):
    print(f.__name__)
    r = range(n)
    t1 = time.perf_counter()
    for i in r:
        f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a)
    t2 = time.perf_counter()
    print((t2-t1)/(10*n))
    return (t2-t1)/(10*n)

def conduct_exp():
    stochtime,classictime = [],[]
    for i in range(20,501,20):
        print("Number of nodes: ",i)
        H = nx.gnm_random_graph(i,i*4)
        G = gt.Graph(directed=False)
        G.add_vertex(n=len(H.nodes()))
        for e in H.edges():
            G.add_edge(e[0],e[1])

        d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)

        stochtime.append(timing(stochastic,3,d))
        classictime.append(timing(classic,3,d))
        print()

    import csv
    with open('data/time_exp1.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(stochtime)
        spamwriter.writerow(classictime)
    csvfile.close()


G = gt.lattice([20,20])

d = distance_matrix.get_distance_matrix(G,'spdm',normalize=False)
classic(d)
#conduct_exp()
