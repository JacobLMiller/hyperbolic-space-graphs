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
import time

def hyperbolic_dist(x1,x2):
    r1,theta1 = x1
    r2,theta2 = x2
    return np.arccosh(np.cosh(r1)*np.cosh(r2)-np.sinh(r1)*np.sinh(r2)*np.cos(theta2-theta1))

def random_point_on_circle(R):
    r = R*pow(random.uniform(0,1),0.5)
    theta = random.uniform(0,2*math.pi)
    return (r,theta)

def init_points(n,r):
    return [(i,{'pos': random_point_on_circle(r)}) for i in range(n)]

def my_output_hyperbolic():
    count = 0
    for i in G.nodes():
        Rh = G.nodes[i]['pos'][0]
        theta = G.nodes[i]['pos'][1]
        Re = (math.exp(Rh)-1)/(math.exp(Rh)+1)

        G.nodes[i]['mypos'] = str(Rh) + "," + str(theta)
        G.nodes[i]['pos'] = str(Re*math.cos(theta)) + "," + str(Re*math.sin(theta))
        count += 1
    nx.drawing.nx_agraph.write_dot(G, "output_hyperbolic.dot")
    nx.drawing.nx_agraph.write_dot(G, "/home/jacob/Desktop/hyperbolic-space-graphs/old/jsCanvas/graphs/hyperbolic_colors.dot")
    nx.drawing.nx_agraph.write_dot(G, "/home/jacob/Desktop/hyperbolic-space-graphs/maps/static/graphs/hyperbolic_colors.dot")


def random_h_graphs_test():
    n = 10
    r = math.log(n)
    print(r)

    G = nx.Graph()
    G.add_nodes_from(init_points(n,r))
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                dist = hyperbolic_dist(G.nodes[i]['pos'],G.nodes[j]['pos'])
                if r-dist > 0:
                    G.add_edge(i,j)
#output_hyperbolic()

def traverse(graph, start, node):
    graph.depth[str(node.name())] = node.shortest_path_distance(start)
    for child in node.hyponyms():
        graph.add_edge(str(node.name()), str(child.name())) # [_add-edge]
        traverse(graph, start, child) # [_recursive-traversal]

def hyponym_graph(start):
    G = nx.Graph() # [_define-graph]
    G.depth = {}
    traverse(G, start, start)
    return G

def compute_scale(scale):
    Y = HMDS(d*scale)
    Y.solve(200)
    return Y.calc_distortion()

def run_HMDS():

    myVar = minimize_scalar(compute_scale,bounds=(0.1,5),method='bounded')
    print(myVar.x)

    Z =HMDS(myVar.x*d)
    Z.solve(1000)
    print(Z.calc_distortion())
    output_hyperbolic(Z.X,G)

def scale_invariance():


    best_score = 10000
    best_X = []

    for i in range(5):
        G = nx.full_rary_tree(2,n)
        d = np.asarray(all_pairs_shortest_path(G))/1

        Y = HMDS(d)
        Y.solve(1000,debug=False)
        print(Y.calc_stress())
        if Y.calc_stress() < best_score:
            best_score = Y.calc_stress()
            best_X = Y.X
            print('got better')
        #print(i)
    output_hyperbolic(best_X,G)

#run_HMDS()

G = nx.drawing.nx_agraph.read_dot('input.dot')
d = np.asarray(all_pairs_shortest_path(G))/1
run_HMDS()

def time():
    start_sgd = time.perf_counter()
    Y = HMDS(d)
    Y.solve(1000)
    Y.calc_stress()
    end_sgd = time.perf_counter()

    start_gd = time.perf_counter()
    Y = MDS(d,geometry='hyperbolic')
    Y.solve(1000)
    Y.calc_stress()
    end_gd = time.perf_counter()

    print(end_sgd-start_sgd)
    print(end_gd-start_gd)
