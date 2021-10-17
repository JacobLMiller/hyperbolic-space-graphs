import random
import math
import networkx as nx

def random_point_on_circle(R,alpha,c):
    r = (1/alpha)*math.acosh(1+(math.cosh(alpha*R)-1)*random.uniform(0,1))
    theta = random.uniform(0,2*math.pi)
    return (r,theta)

def hyperbolic_distance(p1,p2,c):
    r1,theta1 = p1
    r2,theta2 = p2
    return (math.acosh(math.cosh(c*r1)*math.cosh(c*r2) - math.sinh(c*r1)*math.sinh(c*r2)*math.cos(theta2-theta1)))/c

K = -1
c = pow(abs(K),0.5)
R = 5
alpha = 0.8
N = 30
G = nx.Graph()
nodes = []
for i in range(N):
    nodes.append((i, {'pos': random_point_on_circle(R,alpha,c)}))
G.add_nodes_from(nodes)

for i in range(N):
    for j in range(i):
        if R - hyperbolic_distance(nodes[i][1]['pos'],nodes[j][1]['pos'],c) > 0:
            G.add_edge(i,j)

nx.drawing.nx_agraph.write_dot(G, "input.dot")
