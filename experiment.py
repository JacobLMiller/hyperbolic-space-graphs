import igraph as ig
import math
import numpy as np
import random
import networkx as nx
from scipy.optimize import minimize_scalar
import sys
import matplotlib.pyplot as plt
from SGD_MDS import MDS, all_pairs_shortest_path, output_euclidean
from SGD_hyperbolic import HMDS, output_hyperbolic
#from MDS import MDStf, all_pairs_shortest_path
import time
import csv

def output_euclidean(X):
    pos = {}
    count = 0
    for x in G.nodes():
        pos[x] = X[count]
        count += 1
    nx.draw(G,pos=pos)
    plt.show()
    plt.clf()

def collect_data(max_iter,num_of_runs):
    for k in range(2,num_of_runs):
        #G = nx.generators.classic.balanced_tree(r=2,h=k)
        g = ig.Graph.Tree(k,2)
        g.write_dot('input.dot')
        time.sleep(1)

        #G = nx.generators.classic.cycle_graph(10)
        G = nx.drawing.nx_agraph.read_dot('input.dot')
        d = np.asarray(all_pairs_shortest_path(G))/1
        #d = d*2

        #Test traditinoal
        Y = MDS(d)

        start = time.perf_counter()
        Y.solve(num_iter=max_iter,debug=False)
        end = time.perf_counter()

        euclid_time = end-start
        euclid_distortion = get_distortion(Y.X,d)

        #Test Hyperbolic
        Y = HMDS(d)

        start = time.perf_counter()
        Y.solve(num_iter=max_iter,debug=False)
        end = time.perf_counter()

        hyper_time = end-start
        hyper_distortion = Y.calc_distortion()

        #Append fields to csv
        fields = [euclid_time,euclid_distortion,hyper_time,hyper_distortion]
        print("Experiment number: " + str(k) + " out of " + str(num_of_runs))

        with open(r'exp-data-tree2.csv','a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

def geodesic1(xi,xj):
    x1,y1 = xi
    x2,y2 = xj
    return pow(pow(x1-x2,2)+pow(y1-y2,2),0.5)

def geodesic(xi,xj):
    x1,y1 = xi
    x2,y2 = xj
    return np.arccosh(np.cosh(y1)*np.cosh(x2-x1)*np.cosh(y2)-np.sinh(y1)*np.sinh(y2))


def get_stress(X,d,s=1):
    sum = 0
    for i in range(d.shape[0]):
        for j in range(i):
            sum += (1/pow(d[i][j],2))*pow(geodesic(X[i],X[j])-d[i][j],2)
    return pow(sum,0.5)

def get_stress1(X,d,s=1):
    sum = 0
    for i in range(d.shape[0]):
        for j in range(i):
            sum += (1/pow(d[i][j],2))*pow(geodesic1(X[i],X[j])-d[i][j],2)
    return pow(sum,0.5)

def get_distortion(X,d,s=1):
    distortion = 0
    for i in range(d.shape[0]):
        for j in range(i):
            distortion += abs((s*geodesic(X[i],X[j])-s*d[i][j]))/(s*d[i][j])
    return (1/choose(d.shape[0],2))*distortion

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product


max_iter = 1000
num_of_runs = 500
collect_data(max_iter,num_of_runs)

G = nx.drawing.nx_agraph.read_dot("input.dot")
d = np.array(all_pairs_shortest_path(G))


scales = np.linspace(0.5,5,num=50)
hyper_stress = []
stress = []
for i in range(50):
    Y = HMDS(scales[i]*d)
    Y.solve(500)
    hyper_stress.append(get_stress(Y.X,scales[i]*d,1))

    Y = MDS(scales[i]*d)
    Y.solve(500)
    stress.append(get_stress1(Y.X,scales[i]*d,1))

plt.plot(scales, stress, label = "Euclidean MDS")
plt.plot(scales,hyper_stress, label = "Hyperbolic MDS")

plt.xlabel("Scale Factor")
plt.ylabel("Stress")
plt.suptitle("Relationship of stress to scale in Euclidean and hyperbolic space on 5x5 lattice")
plt.legend()


#plt.plot(x, euclid_distortion, label = "Euclidean Distortion")
#plt.plot(x, hyper_distortion, label = "Hyperbolic Distortion")
plt.savefig("figures/scaled_stress.png")

def compute_dist(s):
    G = nx.drawing.nx_agraph.read_dot('input.dot')
    d = np.asarray(all_pairs_shortest_path(G))/1
    Y = HMDS(s*d)
    Y.solve(1000)
    return float(Y.calc_distortion())


#output_euclidean(Y.X)
