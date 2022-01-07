import numpy as np
import graph_tool.all as gt
import pickle
import networkx as nx
from MDS_classic import MDS
from modHMDS import HMDS
import modules.graph_io as graph_io
import modules.distance_matrix as distance_matrix
import time

def classic_exp():
    G = [
        gt.load_graph_from_csv('graphs/data/btree9.txt', csv_options={'delimiter': ' ', 'quotechar': '"'}),
        gt.load_graph_from_csv('graphs/data/qh882.txt', csv_options={'delimiter': ' ', 'quotechar': '"'}),
        gt.load_graph_from_csv('graphs/data/1138_bus.txt', csv_options={'delimiter': ' ', 'quotechar': '"'}),
        gt.load_graph_from_csv('graphs/data/dwt_1005.txt', csv_options={'delimiter': ' ', 'quotechar': '"'})
        ]

    scores = [{} for i in G]

    for i in range(len(G)):
        print(i)

        g = G[i]

        d = distance_matrix.get_distance_matrix(g,distance_metric='spdm',verbose=False,normalize=False)
        print("Graph number: ", i)
        print()

        scores[i] = {
            'stress_hist_classic' : [],
            'stress_hist_stochastic' : [],
            'classic_layout' : [],
            'stochastic_layout' : [],
            'info' : (g,d)
        }

        for j in range(5):
            print("Iteration number: ", j)
            # print("Classic")
            # print()
            # Y = MDS(d,geometry='euclidean')
            # Y.solve(100,debug=True)

            print("Stochastic")
            print()
            Y = myMDS(d)
            Y.solve(3)
            Z = HMDS(d,init_pos=Y.X)
            Z.solve(debug=True)

            #scores[i]['stress_hist_classic'].append(Y.stress_hist)
            scores[i]['stress_hist_stochastic'].append(Z.stress_hist)
            #scores[i]['classic_layout'].append(Y.X)
            scores[i]['stochastic_layout'].append(Z.X)


            with open('data/revision_exp3.pkl', 'wb') as myfile:
                pickle.dump(scores, myfile)
            myfile.close()

        #output_sphere(G,Y.X,'outputs/extend_cube' + str(i) + '.dot')
    with open('data/revision_exp3.pkl', 'wb') as myfile:
        pickle.dump(scores, myfile)

def read_sgd_txt(file):
    f = open(file,"r")
    #splits each line into a list of integers
    lines = [[int(n) for n in x.split()] for x in f.readlines()]
    #closes the file
    f.close()


def testing():
    G = gt.load_graph_from_csv('graphs/data/lesmis.txt', csv_options={'delimiter': ' ', 'quotechar': '"'})
    print(G.num_vertices())
    d = distance_matrix.get_distance_matrix(G,distance_metric='spdm',verbose=False,normalize=False)
    Y = MDS(d,geometry='hyperbolic')
    Y.solve(100,debug=True)
    print(Y.calc_distortion())


def revision_exp3():
    Graphs = [gt.load_graph('graphs/colors.dot'),
              gt.load_graph('graphs/music.dot'),
              gt.load_graph('graphs/btree5.dot'),
              gt.load_graph('graphs/1138_bus.dot')]

    data_error = [[] for g in Graphs]
    data_time = [[] for g in Graphs]
    data = [[] for g in Graphs]
    for g in range(len(Graphs)):
        d = distance_matrix.get_distance_matrix(Graphs[g],distance_metric='spdm',verbose=False,normalize=False)
        for i in range(10):
            start = time.perf_counter()
            Y = HMDS(d)
            Y.solve()
            end = time.perf_counter()

            data_time[g].append(end-start)
            data_error[g].append(Y.calc_distortion())
        print("Error: ",sum(data_error[g])/len(data_error[g]))
        print("Time: ", sum(data_time[g])/len(data_time[g]))
        data[g] = {"time": data_time, "error": data_error}

    import pickle
    with open('data/revision_exp3_sgd.pkl', 'wb') as myfile:
        pickle.dump(data, myfile)
revision_exp3()
