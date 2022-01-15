import numpy as np
import graph_tool.all as gt
import pickle
import networkx as nx
from modHMDS import HMDS
from MDS_classic import MDS
from SGD_MDS import myMDS
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
    #G = [gt.lattice([3,3])]

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

        for j in range(10):
            print("Iteration number: ", j)
            print("Classic")
            print()
            Y = MDS(d,geometry='hyperbolic')
            Y.solve(500,debug=True)
            print(Y.stress_hist[-1])

            print("Stochastic")
            print()
            #Y = myMDS(d)
            #Y.solve(3)
            Z = HMDS(d,init_pos=Y.X)
            Z.solve(100,debug=True)
            print(Z.stress_hist[-1])

            scores[i]['stress_hist_classic'].append(Y.stress_hist)
            scores[i]['stress_hist_stochastic'].append(Z.stress_hist)
            scores[i]['classic_layout'].append(Y.X)
            scores[i]['stochastic_layout'].append(Z.X)


            with open('data/revision_exp1_2.pkl', 'wb') as myfile:
                pickle.dump(scores, myfile)
            myfile.close()

        #output_sphere(G,Y.X,'outputs/extend_cube' + str(i) + '.dot')
    with open('data/revision_exp3.pkl', 'wb') as myfile:
        pickle.dump(scores, myfile)

def random_init_exp():
    G = [
        gt.load_graph_from_csv('graphs/data/btree9.txt', csv_options={'delimiter': ' ', 'quotechar': '"'}),
        gt.load_graph_from_csv('graphs/data/qh882.txt', csv_options={'delimiter': ' ', 'quotechar': '"'}),
        gt.load_graph_from_csv('graphs/data/1138_bus.txt', csv_options={'delimiter': ' ', 'quotechar': '"'}),
        gt.load_graph_from_csv('graphs/data/dwt_1005.txt', csv_options={'delimiter': ' ', 'quotechar': '"'})
        ]
    #G = [gt.lattice([3,3])]

    scores = [{} for i in G]

    for i in range(len(G)):
        print(i)

        g = G[i]

        d = distance_matrix.get_distance_matrix(g,distance_metric='spdm',verbose=False,normalize=False)
        print("Graph number: ", i)
        print()

        scores[i] = {
            'stress_hist_random' : [],
            'stress_hist_smart' : [],
            'random_layout' : [],
            'classic_layout' : [],
            'info' : (g,d)
        }

        for j in range(10):
            print("Iteration number: ", j)
            print("Classic")
            print()
            Y = HMDS(d)
            Y.solve(30,debug=True)

            print("Stochastic")
            print()
            X = myMDS(d)
            X.solve(5)
            Z = HMDS(d,init_pos=X.X)
            Z.solve(30,debug=True)

            scores[i]['stress_hist_random'].append(Y.stress_hist)
            scores[i]['stress_hist_smart'].append(Z.stress_hist)
            scores[i]['random_layout'].append(Y.X)
            scores[i]['classic_layout'].append(Z.X)


            with open('data/revision_exp_smart_init.pkl', 'wb') as myfile:
                pickle.dump(scores, myfile)
            myfile.close()

        #output_sphere(G,Y.X,'outputs/extend_cube' + str(i) + '.dot')
    with open('data/revision_exp_smart_init.pkl', 'wb') as myfile:
        pickle.dump(scores, myfile)

def read_sgd_txt(file):
    f = open(file,"r")
    #splits each line into a list of integers
    lines = [[int(n) for n in x.split()] for x in f.readlines()]
    #closes the file
    f.close()


def testing():
    # G = gt.load_graph_from_csv('graphs/data/lesmis.txt', csv_options={'delimiter': ' ', 'quotechar': '"'})
    # print(G.num_vertices())
    # d = distance_matrix.get_distance_matrix(G,distance_metric='spdm',verbose=False,normalize=False)
    # Y = MDS(d,geometry='hyperbolic')
    # Y.solve(100,debug=True)
    # print(Y.calc_distortion())

    G = gt.load_graph('SGD/graphs/colors.dot')
    d = distance_matrix.get_distance_matrix(G,distance_metric='spdm',verbose=False,normalize=False)
    Y = HMDS(d)
    Y.solve()
    import networkx as nx
    G = nx.drawing.nx_agraph.read_dot('SGD/graphs/colors.dot')
    from modHMDS import output_hyperbolic
    output_hyperbolic(Y.X,G)


def revision_exp3():
    Graphs = [gt.load_graph('graphs/colors.dot'),
              gt.load_graph('graphs/music.dot'),
              gt.load_graph('graphs/btree8.dot'),
              gt.load_graph('graphs/1138_bus.dot')]

    data_error = [[] for g in Graphs]
    data_time = [[] for g in Graphs]
    data = [[] for g in Graphs]
    for g in range(len(Graphs)):

        for i in range(10):
            start = time.perf_counter()
            d = distance_matrix.get_distance_matrix(Graphs[g],distance_metric='spdm',verbose=False,normalize=False)
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


def read_data():
    import pickle
    with open("data/exp3/projection_error_colors.pkl",'rb') as myfile:
        time = pickle.load(myfile)
    data = np.array(time)
    print(data.mean())



#classic_exp()
#G = gt.load_graph_from_csv('SGD/graphs/data/qh882.txt', csv_options={'delimiter': ' ', 'quotechar': '"'})
#G = gt.load_graph('SGD/graphs/small_block.dot')
G = gt.lattice([10,10])
print(G.num_vertices())

start = time.perf_counter()
d = distance_matrix.get_distance_matrix(G,distance_metric='spdm',verbose=False,normalize=False)
Z = HMDS(d)
Z.solve(25,debug=False)

print((time.perf_counter()-start)/10)
# X = np.zeros(Z.X.shape)
# count = 0
# import math
# for x,y in Z.X:
#     Rh = x
#     theta = y
#     Rh = np.arccosh(np.cosh(x)*np.cosh(y))
#     theta = 2*math.atan2(np.sinh(x)*np.cosh(y)+pow(pow(np.cosh(x),2)*pow(np.cosh(y),2)-1,0.5),np.sinh(y))
#     Re = (math.exp(Rh)-1)/(math.exp(Rh)+1)
#     #hR = math.acosh((r*r/2)+1)
#     X[count] = np.array([Rh,theta])
#     count += 1
#
#
pos = G.new_vp('vector<float>')
pos.set_2d_array(Z.X.T)
G.vertex_properties['pos'] = pos

for v in  G.iter_vertices():
    print(v)
    print(Z.X[int(v)])

import io
import tempfile

with tempfile.TemporaryFile() as file:
    G.save(file,fmt='dot')
    file.seek(0)
    dot_rep = file.read()
print(dot_rep)

def gt_to_json(G,embedding):
    nodes, edges = G.iter_vertices(),G.iter_edges()

    out = {'nodes': [None for i in range(G.num_vertices())],
            'edges': [None for i in range(G.num_edges())]
            }
    for v in nodes:
        out['nodes'][int(v)] = {
            'id': int(v),
            'pos': list(embedding[int(v)])
        }
    count = 0
    for u,v in edges:
        ##Implement map or zip or something
        out['edges'][count] = {
            's': int(u),
            't': int(v)
        }
        count += 1

    return out

json_rep = gt_to_json(G,Z.X)
print(json_rep)
import json
X = json.dumps(json_rep)
print(type(X))
# G.save('test.dot')
# G.save("/home/jacob/Desktop/hyperbolic-space-graphs/old/jsCanvas/graphs/hyperbolic_colors.dot")
# G.save("/home/jacob/Desktop/hyperbolic-space-graphs/maps/static/graphs/hyperbolic_colors.dot")
