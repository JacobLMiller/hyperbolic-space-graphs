import math
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
from SGD_hyperbolic import HMDS,all_pairs_shortest_path,output_hyperbolic
from MDS import MDS
from nltk.corpus import wordnet as wn
from scipy.optimize import minimize_scalar
import igraph as ig
from euclid_random_graph import generate_graph
import csv
import time

def compute_scale(scale):
    Y = HMDS(d,epsilon=scale)
    Y.solve(25)
    return Y.calc_stress()

def run_HMDS():

    myVar = minimize_scalar(compute_scale,bounds=(1e-3,10),method='bounded')
    print(myVar.x)

    Z = HMDS(d,epsilon=myVar.x)
    Z.solve(25)
    print(Z.calc_distortion())
    output_hyperbolic(Z.X,G)

def scale_test():
    scale = np.array([0.05,0.1,0.15,0.2])

    n = 50
    p = 0.6
    trials = {}

    #G = nx.triangular_lattice_graph(2,2)
    G = nx.full_rary_tree(2,20)
    d = np.asarray(all_pairs_shortest_path(G))

    S = MDS(d,geometry="hyperbolic")

    for s in scale:
        print(s)
        trials[str(s)] = 0
        for i in range(25):
            G = nx.erdos_renyi_graph(30,0.5)
            d = np.asarray(all_pairs_shortest_path(G))
            Z = HMDS(d,epsilon=s)
            Z.solve(15)
            trials[str(s)] += Z.calc_stress()

        trials[str(s)] = trials[str(s)]/25


    with open('data/optimize-eta.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for s in trials:
            spamwriter.writerow([str(s)] + [str(trials[str(s)])])

def classicTrial(d):
    Y = MDS(d,geometry='hyperbolic')
    start = time.perf_counter()
    Y.solve(100)
    end = time.perf_counter()
    return end-start

def stochasticTrial(d):
    Y = HMDS(d)
    start = time.perf_counter()
    Y.solve(100)
    end = time.perf_counter()
    return end-start

def time_test():
    trials = [n for n in range(10,100)]
    classicTime = [0 for i in range(10,100)]
    stochasticTime = [0 for i in range(10,100)]
    for t in range(len(trials)):
        print(t)
        G = nx.erdos_renyi_graph(trials[t],0.5)
        d = np.array(all_pairs_shortest_path(G))/1

        classicTime[t] = classicTrial(d)
        stochasticTime[t] = stochasticTrial(d)

    with open('data/time_trials.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(trials)):
            spamwriter.writerow([str(classicTime[i])] + [str(stochasticTime[i])])
#Z = HMDS(d)
#Z.solve(1000)
#print(Z.calc_distortion())
#output_hyperbolic(Z.X,G)
#G = nx.drawing.nx_agraph.read_dot('input.dot')
G = nx.full_rary_tree(2,30)
d = np.asarray(all_pairs_shortest_path(G))/1

run_HMDS()
#time_test()

#scale_test()
