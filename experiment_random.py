import math
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import csv

from randomizationHMDS import HMDS, all_pairs_shortest_path
from modHMDS import myHMDS, output_hyperbolic
from SGD_MDS import myMDS

from annealingTest import HMDS_step

from hyper_random_graph import get_hyperbolic_graph


def run_random():
    G = [nx.drawing.nx_agraph.read_dot('SGD/input.dot')]

    rr = []
    indices = []
    replacement = []

    count = 0

    for g in G:
        d = np.array(all_pairs_shortest_path(g))/1

        print(count)
        temp = 0

        for i in range(50):
            temp += np.array(HMDS(d).solve_rr(1000,debug=True))
        rr.append(temp/50)

        temp = 0

        for i in range(50):
            temp += np.array(HMDS(d).solve_shuffle_indices(1000,debug=True))
        indices.append(temp/50)


        temp = 0

        for i in range(50):
            temp += np.array(HMDS(d).solve_sample_replacement(1000,debug=True))
        replacement.append(temp/50)


        count += 1
    data = [rr,indices,replacement]

    with open('data/randomization_colors.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(3):
            for j in range(len(G)):
                spamwriter.writerow([str(k) for k in data[i][j]])

def run_num_iterations():
    #G = nx.drawing.nx_agraph.read_dot('SGD/input.dot')
    G = [nx.random_tree(50),get_hyperbolic_graph(50),nx.les_miserables_graph(),nx.drawing.nx_agraph.read_dot('SGD/input.dot')]
    data = [i for i in G]
    count = 0

    for g in G:
        d = np.array(all_pairs_shortest_path(g))/1
        loss = np.zeros(100)
        for i in range(10):
            loss += np.array(HMDS_step(d).solve(100,debug=True,step_type='sqrt'))
        loss = loss/10

        data[count] = loss
        count += 1
        #plt.show()

    with open('data/num_iterations2.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in range(len(G)):
            spamwriter.writerow([str(k) for k in data[i]])

def run_epsilon():
    G = nx.drawing.nx_agraph.read_dot('input.dot')
    d = np.array(all_pairs_shortest_path(G))/1
    eps = np.linspace(0.05,0.2,num=10)

    for i in range(len(eps)):
        print(eps[i])
        Y = myHMDS(d,epsilon=eps[i])
        Y.solve(30)
        print(Y.calc_stress())

def run_init():
    #G = nx.drawing.nx_agraph.read_dot('SGD/input.dot')
    G = [nx.grid_graph([10,10]),nx.random_tree(50),get_hyperbolic_graph(50),nx.les_miserables_graph(),nx.drawing.nx_agraph.read_dot('SGD/input.dot')]
    #d = np.array(all_pairs_shortest_path(G))/1

    random_stresses = [i for i in G]
    smart_stresses = [i for i in G]

    count = 0

    for g in G:
        d = np.array(all_pairs_shortest_path(g))/1

        random_stress = [i for i in range(25)]
        for j in range(25):
            stress = []
            for i in range(5):
                Y = myHMDS(d)
                Y.solve(15)
                stress.append(Y.calc_stress())
            stress = sum(stress)/5
            random_stress[j] = stress
            print(stress)
        random_stresses[count] = sum(random_stress)/len(random_stress)

        smart_stress = [i for i in range(25)]

        for j in range(25):
            Z = myMDS(d)
            Z.solve(15)

            Y = myHMDS(d,init_pos=Z.X)
            Y.solve(15)
            stress = Y.calc_stress()
            print(stress)
            smart_stress[j] = stress
        smart_stresses[count] = sum(smart_stress)/len(smart_stress)

        count += 1

    with open('data/init_test.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(G)):
            spamwriter.writerow([str(random_stresses[i])])
            spamwriter.writerow([str(smart_stresses[i])])


def run_schedule():
    G = [nx.grid_graph([10,10]),nx.random_tree(50),get_hyperbolic_graph(50),nx.les_miserables_graph(),nx.drawing.nx_agraph.read_dot('SGD/input.dot')]
    d = np.array(all_pairs_shortest_path(G[0]))/1

    exponential = []
    fraction = []
    sqrt = []

    count = 0

    for g in G:
        d = np.array(all_pairs_shortest_path(g))/1

        print(count)
        temp = 0

        #for i in range(15):
        #    temp += np.array(HMDS_step(d).solve(500,debug=True,step_type='default'))
        #exponential.append(temp/15)

        #temp = 0

        #for i in range(15):
        #    temp += np.array(HMDS_step(d).solve(500,debug=True,step_type='fraction'))
        #fraction.append(temp/15)


        temp = 0

        for i in range(15):
            temp += np.array(HMDS_step(d).solve(40,debug=True,step_type='sqrt'))
        sqrt.append(temp/15)


        count += 1

    data = [sqrt]

    with open('data/annealing_schedule4.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(1):
            for j in range(len(G)):
                spamwriter.writerow([str(k) for k in data[i][j]])

def tree_test():
    euclidean = []
    hyperbolic = []
    for i in range(10,201,10):
        G = nx.full_rary_tree(2,i)
        d = np.array(all_pairs_shortest_path(G))/1

        loss_Euclidean = 0
        loss_Hyperbolic = 0

        for j in range(25):

            Y = myMDS(d)
            Y.solve(15)
            loss_Euclidean += Y.calc_distortion()
            Z = HMDS_step(d)
            Z.solve(20)
            loss_Hyperbolic += Z.calc_distortion()

        loss_Euclidean = loss_Euclidean/25
        loss_Hyperbolic = loss_Hyperbolic/25

        euclidean.append(loss_Euclidean)
        hyperbolic.append(loss_Hyperbolic)

    data = [euclidean,hyperbolic]

    with open('data/trees2.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(2):
            spamwriter.writerow([str(k) for k in data[i]])





tree_test()
#run_schedule()
#run_random()
#run_init()
#run_num_iterations()
#run_epsilon()
G = nx.drawing.nx_agraph.read_dot('SGD/input.dot')
G = nx.full_rary_tree(2,50)

#G = nx.grid_graph([10,10])
#d = np.array(all_pairs_shortest_path(G))/1
#print(np.max(d))

#Y = HMDS_step(d)
#Y.solve(50,debug=True,epsilon=0.01)
#print(Y.calc_distortion())
