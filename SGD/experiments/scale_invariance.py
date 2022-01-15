import math
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
from SGD_hyperbolic import HMDS,all_pairs_shortest_path,output_hyperbolic
from SGD_MDS import myMDS
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
    avgStress = np.zeros(scale.shape)

    n = 50
    p = 0.6
    trials = np.zeros(25)

    #G = nx.triangular_lattice_graph(2,2)
    G = nx.full_rary_tree(2,20)
    d = np.asarray(all_pairs_shortest_path(G))

    S = MDS(d,geometry="hyperbolic")

    for s in range(len(scale)):
        print(s)
        for i in range(len(trials)):
            G = nx.grid_graph([5,5])
            d = np.asarray(all_pairs_shortest_path(G))
            Z = HMDS(d,epsilon=scale[s])
            Z.solve(15)
            trials[i] = Z.calc_stress()

        avgStress[s] = trials.mean()


    with open('data/optimize-eta.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for s in range(len(scale)):
            spamwriter.writerow([str(scale[s])] + [str(avgStress[s])])

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
        for i in range(len()):
            spamwriter.writerow([str(classicTime[i])] + [str(stochasticTime[i])])

def tree_test():
    ns = [i for i in range(10,100)]
    trials = 10
    print(ns[10])

    euclideanTrees = []
    hyperTrees = []

    for i in range(len(ns)):
        print(i)
        average = 0
        for j in range(trials):
            G = nx.random_tree(n=ns[i])
            d = np.asarray(all_pairs_shortest_path(G))/1

            Y = myMDS(d)
            Y.solve(15)
            average += Y.calc_distortion()
        average = average/trials
        euclideanTrees.append(average)

        average = 0
        for j in range(trials):
            G = nx.random_tree(n=ns[i])
            d = np.asarray(all_pairs_shortest_path(G))/1

            Z = myMDS(d)
            Z.solve(15)
            init = np.ones(Z.X.shape)
            for k in range(len(Z.X)):
                r = pow(pow(Z.X[k][0],2)+pow(Z.X[k][1],2),0.5)
                theta = math.atan2(Z.X[k][1],Z.X[k][0])
                init[k] = np.array([r,theta])

            Y = HMDS(d,init_pos=init)
            Y.solve(15)
            average += Y.calc_distortion()
        average = average/trials
        hyperTrees.append(average)

    with open('data/time_trials.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(ns)):
            spamwriter.writerow([str(ns[i])] + [str(euclideanTrees[i])] + [str(hyperTrees[i])])

#Z = HMDS(d)
#Z.solve(1000)
#print(Z.calc_distortion())
#output_hyperbolic(Z.X,G)
#G = nx.drawing.nx_agraph.read_dot('input.dot')
#G = nx.full_rary_tree(2,30)
#d = np.asarray(all_pairs_shortest_path(G))/1

#run_HMDS()
#time_test()
tree_test()
#scale_test()
