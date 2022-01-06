import math
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
#from SGD_hyperbolic import HMDS,all_pairs_shortest_path,output_hyperbolic
from modHMDS import HMDS, output_hyperbolic
from SGD_MDS import myMDS,all_pairs_shortest_path
from MDS import MDS
from nltk.corpus import wordnet as wn
from scipy.optimize import minimize_scalar
import igraph as ig
from euclid_random_graph import generate_graph
import csv
import time
from hyper_random_graph import get_hyperbolic_graph

def tree_test():
    ns = [i for i in range(10,500)]
    trials = 10
    print(ns[10])

    euclideanTrees = []
    hyperTrees = []

    for i in range(len(ns)):
        print(i)

        g = ig.Graph.Tree(ns[i],2)
        g.write_dot('input.dot')
        time.sleep(1)

        G = nx.drawing.nx_agraph.read_dot('input.dot')
        d = np.array(all_pairs_shortest_path(G))/1

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

            Y = HMDS(d)
            Y.solve(25)
            average += Y.calc_distortion()
        average = average/trials
        hyperTrees.append(average)

    with open('data/tree_trials.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(ns)):
            spamwriter.writerow([str(ns[i])] + [str(euclideanTrees[i])] + [str(hyperTrees[i])])

def time_test():
    ns = [i for i in range(10,100)]
    trials = 10
    print(ns[10])

    classicTime = []
    stochasticTime = []

    for i in range(len(ns)):
        print(i)

        G = nx.erdos_renyi_graph(ns[i],0.5)
        d = np.asarray(all_pairs_shortest_path(G))/1

        start = time.perf_counter()
        Y = MDS(d,geometry='hyperbolic')
        Y.solve(25)
        end = time.perf_counter()
        classicTime.append(end-start)

        start = time.perf_counter()
        Y = HMDS(d)
        Y.solve(25)
        end = time.perf_counter()
        stochasticTime.append(end-start)

    with open('data/time_trials.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(ns)):
            spamwriter.writerow([str(ns[i])] + [str(classicTime[i])] + [str(stochasticTime[i])])

def random_test():
    ns = [i for i in range(10,100)]
    trials = 10
    print(ns[10])

    euclideanTrees = []
    hyperTrees = []

    for i in range(len(ns)):
        print(i)
        average = 0
        for j in range(trials):
            G = get_hyperbolic_graph(n=ns[i])
            d = np.asarray(all_pairs_shortest_path(G))/1

            Y = myMDS(d)
            Y.solve(25)
            average += Y.calc_distortion()
        average = average/trials
        euclideanTrees.append(average)

        average = 0
        for j in range(trials):
            G = get_hyperbolic_graph(n=ns[i])
            d = np.asarray(all_pairs_shortest_path(G))/1

            Y = HMDS(d)
            Y.solve(25)
            average += Y.calc_distortion()
        average = average/trials
        hyperTrees.append(average)

    with open('data/random_trials.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(ns)):
            spamwriter.writerow([str(ns[i])] + [str(euclideanTrees[i])] + [str(hyperTrees[i])])

#tree_test()
#random_test()
time_test()
