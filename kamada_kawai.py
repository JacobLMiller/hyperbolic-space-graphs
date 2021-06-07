import networkx as nx
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

import math
import random
import time

from networkx.drawing.nx_agraph import write_dot


def bfs(G,start):
    queue = [start]
    discovered = [start]
    distance = {start: 0}

    while len(queue) > 0:
        v = queue.pop()

        for w in G.neighbors(v):
            if w not in discovered:
                discovered.append(w)
                distance[w] =  distance[v] + 1
                queue.insert(0,w)

    myList = []
    for x in G.nodes:
        myList.append(distance[x])

    return myList

def all_pairs_shortest_path(G):
    d = [ [ -1 for i in range(len(G.nodes)) ] for j in range(len(G.nodes)) ]

    count = 0
    for node in G.nodes:
        d[count] = bfs(G,node)
        count += 1
    return d

def compute_l_table(d):
    maximum = 0
    for x in d:
        for y in x:
            maximum = max(maximum,y)
    L0 = 1
    L = L0/maximum


    l = [ [ -1 for i in range(len(G.nodes)) ] for j in range(len(G.nodes)) ]
    for x in range(len(d)):
        for y in range(len(d[x])):
            l[x][y] = L*d[x][y]


    return l

def compute_k(d):
    K = 1

    k = [ [ 10000 for i in range(len(G.nodes)) ] for j in range(len(G.nodes)) ]
    for x in range(len(d)):
        for y in range(len(d[x])):
            if x == y:
                k[x][y] = 0
            else:
                k[x][y] = K/(d[x][y]**2)

    return k


def parse_pos(str_pos):
    return (float(str_pos.split(",")[0]),float(str_pos.split(",")[1]))

def dEdP(G,m,tables):

    x = 0
    y = 0

    for i in G.nodes:
        if i == m:
            continue



        xDif = G.nodes[m]['pos'][0] - G.nodes[i]['pos'][0]

        yDif = G.nodes[m]['pos'][1] - G.nodes[i]['pos'][1]
        xDif2 = xDif*xDif

        yDif2 = yDif*yDif
        denominator = (xDif2 + yDif2)**(1/2)


        k_mi = tables['k'][G.nodes[m]['index']][G.nodes[i]['index']]


        numerator = tables['l'][G.nodes[m]['index']][G.nodes[i]['index']]*xDif


        division = numerator/denominator


        x = x + k_mi*(xDif - division)



        numerator = tables['l'][G.nodes[m]['index']][G.nodes[i]['index']]*yDif
        division = numerator/denominator
        y += k_mi*(yDif - division)



    return np.array([x,y])

def dE2(G,m,tables):
    x = 0
    y = 0
    xy = 0
    yx = 0

    for i in G.nodes:
        if i == m:
            continue
        xDif = xDif = G.nodes[m]['pos'][0] - G.nodes[i]['pos'][0]
        yDif = G.nodes[m]['pos'][1] - G.nodes[i]['pos'][1]
        k_mi = tables['k'][G.nodes[m]['index']][G.nodes[i]['index']]
        l_mi = tables['l'][G.nodes[m]['index']][G.nodes[i]['index']]
        denominator =(xDif**2 + yDif**2)**(3/2)

        numerator = l_mi*(yDif)**2
        fraction = numerator/denominator
        x += k_mi*(1-fraction)

        numerator = l_mi*xDif*yDif
        fraction = numerator/denominator
        xy += k_mi*fraction

        numerator = l_mi*(xDif)**2
        fraction = numerator/denominator
        y += k_mi*(1-fraction)


    return np.array([[x,xy],[xy,y]])

def compute_delta_x_and_y(G,p,tables):

    coefficientMatrix = dE2(G,p[0],tables)

    answerMatrix = -1*dEdP(G,p[0],tables)


    return np.linalg.inv(coefficientMatrix).dot(answerMatrix)

def find_max_delta(G,tables):
    deltas = {}
    max_delta = ('dummy',-1)
    for m in G.nodes:
        partialDerivative = dEdP(G,m,tables)
        deltas[m] = math.sqrt(partialDerivative[0]**2 + partialDerivative[1]**2)

        if max_delta[1] < deltas[m]:
            max_delta = (m,deltas[m])
    print("max_delta = " + str(max_delta))
    return max_delta

def generate_N_polygon(N):
    verticesList = []
    theta = (2*math.pi)/N
    R = 10
    for i in range(N):
        verticesList.append((R+math.sin(i*theta),R+math.cos(i*theta)))

    return (verticesList)

def computeKKlayout(G):
    D = all_pairs_shortest_path(G)
    L = compute_l_table(D)
    K = compute_k(D)
    tables = {'d': D,'l': L, 'k': K}

    #Initialize node postions
    count = 0
    nPolyVertices = generate_N_polygon(len(G.nodes))
    for i in G.nodes:
        G.nodes[i]['pos'] = nPolyVertices[count]
        G.nodes[i]['index'] = count
        count += 1

    p = find_max_delta(G,tables)

    count = 0
    prev = 0
    bigCount = 0
    keepGoing = True
    while (keepGoing):

        count1 = 0

        deltaP = compute_delta_x_and_y(G,p,tables)
        #print(deltaP)

        x = G.nodes[p[0]]['pos'][0] + deltaP[0]
        y = G.nodes[p[0]]['pos'][1] + deltaP[1]

        G.nodes[p[0]]['pos'] = (x,y)

        if count1 % 100 == 0:
            pass
            #print(x)

        #G.nodes[p[0]]['pos'] = (x,y)
        #p_d = dEdP(G,p[0],tables)
        #p = (p[0], math.sqrt(p_d[0]**2 + p_d[1]**2))
        #print(p)


        p = find_max_delta(G,tables)

        epsilon = p[1] - prev
        if abs(epsilon) < .001:
            bigCount += 1
        if bigCount > 10:
            break

        prev = p[1]

        count += 1
        if count > 10000:
            break



    return G

def tuple_to_string(convert_tuple):
    string1 = str(convert_tuple[0])
    string2 = str(convert_tuple[1])
    return(string1 + "," + string2)



g = ig.Graph.Tree(25, 3)
g.write_dot('lattice_graph.dot')

time.sleep(1)


G = nx.drawing.nx_agraph.read_dot('jsCanvas/graphs/ring.dot')
#G = nx.tetrahedral_graph()
G = computeKKlayout(G)


posDict = {}
for i in G.nodes:
    print(G.nodes[i])
    #G.nodes[i]['pos'] = tuple_to_string(G.nodes[i]['pos'])
    posDict[i] = G.nodes[i]['pos']
    G.nodes[i]['pos'] = tuple_to_string(G.nodes[i]['pos'])


write_dot(G, "jsCanvas/graphs/normal_colors.dot")
print("Completed")


nx.draw(G,posDict)  # networkx draw()

plt.draw()  # pyplot draw()
plt.show()
