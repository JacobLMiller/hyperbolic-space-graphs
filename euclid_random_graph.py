import random
import math
import networkx as nx

def random_point_on_circle(R,alpha):
    r = R*pow(random.uniform(0,1),0.5)
    theta = random.uniform(0,2*math.pi)
    return (r*math.cos(theta),r*math.sin(theta))

def hyperbolic_distance(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return pow(pow(x2-x1,2)+pow(y2-y1,2),0.5)

def generate_graph(n, p):
    R = 1
    alpha = 2
    N = n
    G = nx.Graph()
    nodes = []
    for i in range(N):
        nodes.append((i, {'pos': random_point_on_circle(R,alpha)}))
    G.add_nodes_from(nodes)

    for i in range(N):
        for j in range(i):
            if hyperbolic_distance(nodes[i][1]['pos'],nodes[j][1]['pos']) < p:
                G.add_edge(i,j)

    nx.drawing.nx_agraph.write_dot(G, "input.dot")
    return G
#generate_graph(100,0.3)
