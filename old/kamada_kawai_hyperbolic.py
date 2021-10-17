import networkx as nx
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import drawSvg as draw

import math
import random
import cmath
import copy
import time
import os

from networkx.drawing.nx_agraph import write_dot
from drawSvg import Drawing
from hyperbolic import euclid, util
from hyperbolic.poincare.shapes import *
from hyperbolic.poincare import Transform
from shutil import rmtree

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


    l = [ [ -1 for i in range(len(d)) ] for j in range(len(d)) ]
    for x in range(len(d)):
        for y in range(len(d[x])):
            l[x][y] = L*d[x][y]


    return l

def compute_k(d):
    K = 2

    k = [ [ 10000 for i in range(len(d)) ] for j in range(len(d)) ]
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
        #print(yDif)

        xDif2 = xDif*xDif


        yDif2 = yDif*yDif
        denominator = (xDif2 + yDif2)**(1/2)


        if(denominator == 0):
            print(G.nodes[i]['pos'])
            print(G.nodes[m]['pos'])
            print(i)
            print(m)

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
        xDif = G.nodes[m]['pos'][0] - G.nodes[i]['pos'][0]
        yDif = G.nodes[m]['pos'][1] - G.nodes[i]['pos'][1]
        #print(yDif)



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


    coefficientMatrix = dE2(G,p,tables)


    answerMatrix = -1*dEdP(G,p,tables)



    return np.linalg.inv(coefficientMatrix).dot(answerMatrix)

def find_max_delta(G,tables):
    deltas = {}
    max_delta = ('dummy',-1)
    for m in G.nodes:
        H = tau(G,m)

        partialDerivative = dEdP(G,m,tables)
        deltas[m] = math.sqrt(partialDerivative[0]**2 + partialDerivative[1]**2)

        if max_delta[1] < deltas[m]:
            max_delta = (m,deltas[m])
    print("max_delta = " + str(max_delta))
    return max_delta

def generate_N_polygon(N):
    verticesList = []
    theta = (2*math.pi)/N
    R = 1
    for i in range(N):
        verticesList.append((R+math.sin(i*theta)-.5,R+math.cos(i*theta)-.5))

    return verticesList

def getPolar(point):
    x = point[0]
    y = point[1]

    r = math.sqrt(x**2 + y**2)
    theta = 0
    if y >= 0:
        theta = math.acos(x/r)
    else:
        theta = -1*math.acos(x/r)
    return(r,theta)

def mobius(z,a,b,c,d):
    return (a*z + b)/(c*z + d)

def tau(G,p):
    H = copy.deepcopy(G)
    z0 = complex(*G.nodes[p]['pos'])

    for z in H.nodes:
        if H.nodes[z]['pos'] == H.nodes[p]['pos']:
            #temp  = mobius(z0,1,-z0,(-z0).conjugate(),1)
            H.nodes[z]['pos'] = (0,0)
            continue

        H.nodes[z]['pos'] = mobius(complex(*H.nodes[z]['pos']),1,-z0,(-z0).conjugate(),1)


        norm = cmath.polar(H.nodes[z]['pos'])[0]
        #if(norm == 1):
        #    norm = norm - .01
        if(norm >= 1):
            norm = .999
        if(norm == 0):
            norm = .001
            H.nodes[z]['pos'] = complex(0.001,0.001)

        #numerator = 1 + norm
        #denominator = 1 - norm
        #division = numerator/denominator
        #term2 = np.log(division)

        #print(norm)
        term2 = 2*math.atanh(norm)

        term1 = H.nodes[z]['pos']/norm
        temp = term1*term2
        if(norm == .001):
            print(term1)
            print(term2)
            print(H.nodes[z]['pos'])
        H.nodes[z]['pos'] = (temp.real,temp.imag)
        #print("The position in G' is " + str(H.nodes[z]['pos']))

    return H




def inverseTau(z,z0,Y):

    norm = abs(z)
    if(norm > 100):
        print(norm)
        print(z)
        print(Y.nodes)
    numerator = 1-(np.e**norm)
    denominator = 1+(np.e**norm)
    division = abs(numerator/denominator)

    term1 = z/norm
    newZ = term1*division
    #print("newZ is " + str(newZ))


    finalZ = mobius(newZ,-1,-z0,(-z0).conjugate(),-1)

    #print("transform  is " + str((-z0,(-z0).conjugate())))
    #print()

    return(finalZ.real,finalZ.imag)


def azimuthal(p):
    r = abs(complex(*p))
    theta = 0
    if p[1] >= 0:
        theta = math.acos(p[0]/r)
    else:
        theta = -1*math.acos(p[0]/r)
    hR = math.acosh((r*r + 2)/2)

    if hR < 0:
        theta += math.pi
        hR = abs(hR)

    if theta < 0:
        theta = abs(math.floor(theta / math.tau)) * math.tau + theta
    elif theta >= math.tau:
        theta = theta % math.tau

    er = (math.exp(hR)-1)/(math.exp(hR)+1)
    x = er*math.cos(theta)
    y = er*math.sin(theta)

    return(x,y)

def Draw_SVG(G,number):
    points = []
    lines = []
    nodeDict = {}
    d = Drawing(2.1,2.1, origin='center')
    d.draw(euclid.shapes.Circle(0, 0, 1), fill='#ddd')

    for i in G.nodes:
        #print(G.nodes[i]['pos'])
        #print(cmath.polar(complex(*G.nodes[i]['pos'])))
        points.append(Point(G.nodes[i]['pos'][0],G.nodes[i]['pos'][1]))
        nodeDict[i] = points[-1]

    for i in G.edges:
        lines.append(Line.fromPoints(*nodeDict[i[0]],*nodeDict[i[1]],segment=True))

    trans = Transform.shiftOrigin(points[0])

    for i in lines:
        d.draw(i,transform=trans,hwidth=.01,fill='black')

    for i in points:
        d.draw(i,transform=trans,hradius=.05,fill='green')


    d.setRenderSize(w=400)
    d.saveSvg('slideshow/Test' + str(number) + '.svg')

def compute_geometric_mean(G):
    count = 0
    allX = 0
    allY = 0

    for x in G.nodes:
        pos = G.nodes[x]['pos']
        allX += pos[0]
        allY += pos[1]
        count += 1

    return (allX/count,allY/count)

def computeKKlayout(X):
    #rmtree('slideshow')

    #rowser-based Hyperbolic Visualization of Graphsos.mkdir('slideshow')

    D = all_pairs_shortest_path(X)
    L = compute_l_table(D)
    K = compute_k(D)
    tables = {'d': D,'l': L, 'k': K}
    for x in L:
        print(x)
    print()

    #Initialize node postions
    count = 0
    nPolyVertices = generate_N_polygon(len(X.nodes))
    for i in X.nodes:
        X.nodes[i]['pos'] = azimuthal(nPolyVertices[count])
        X.nodes[i]['index'] = count
        count += 1

    p = find_max_delta(X,tables)
    #p = ["0",100]
    count = 0
    count2 = 0
    bigCount = 0
    prev = 0
    keepGoing = True
    while (keepGoing):

        count1 = 0
        i = p[0]

        #i = p[0]
        z0 = complex(*X.nodes[i]['pos'])
        #print("z0 is " + str(z0))


        H = tau(X,i)
        #for x in H.nodes:
        #    print(H.nodes[x]['pos'])
        #print()


        deltaP = compute_delta_x_and_y(H,i,tables)


        #print('current position:' + str(X.nodes[i]['pos']))

        x = deltaP[0]
        y = deltaP[1]

        #print(X.nodes[i])

        X.nodes[i]['pos'] = inverseTau(complex(*(x,y)),z0,X)
        #print("New position: " + str(X.nodes[i]['pos']))
        #print("--------------")
        #print()

        #if count1 % 10 == 0:
            #print(x)

        #X.nodes[p[0]]['pos'] = (x,y)
        #p_d = dEdP(X,p[0],tables)
        #p = (p[0], math.sqrt(p_d[0]**2 + p_d[1]**2))
        #print(p)



        p = find_max_delta(X,tables)
        #print(X.nodes['grey']['pos'])



        #Draw_SVG(X,count2)
        count2 += 1

        epsilon = p[1] - prev
        #if abs(epsilon) < .001:
        #    bigCount += 1
        #if bigCount > 10:
        #    break

        prev = p[1]

        count += 1
        if count > 1000:
            break



    return X

def tuple_to_string(convert_tuple):
    string1 = str(convert_tuple[0])
    string2 = str(convert_tuple[1])
    return(string1 + "," + string2)



g = ig.Graph.GRG(50,.3)
g.write_dot('hyperbolic_colors.dot')



#X = nx.drawing.nx_agraph.read_dot('jsCanvas/graphs/hyperbolic_colors.dot')
#write_dot(X, "jsCanvas/graphs/hyperbolic_colors.dot")
#X = nx.drawing.nx_agraph.read_dot('jsCanvas/graphs/ring.dot')
#print(X.nodes)

G = nx.hexagonal_lattice_graph(5,5)
G = computeKKlayout(G)

#Draw_SVG(G,1000)

#mean = compute_geometric_mean(G)
p = 0
for i in G.nodes:
    p = i
    break

z0 = complex(*G.nodes[p]['pos'])
for i in G.nodes:
    pos = mobius(complex(*G.nodes[i]['pos']),1,-z0,-z0.conjugate(),1)
    #G.nodes[i]['pos'] = (pos[0]-mean[0],pos[1]-mean[1])
    G.nodes[i]['pos'] = (pos.real,pos.imag)
    print(G.nodes[i]['pos'])



posDict = {}
for i in G.nodes:
    print(G.nodes[i])
    #G.nodes[i]['pos'] = tuple_to_string(G.nodes[i]['pos'])
    posDict[i] = G.nodes[i]['pos']
    G.nodes[i]['mypos'] = tuple_to_string(G.nodes[i]['pos'])


nx.drawing.nx_agraph.write_dot(G, "old/jsCanvas/graphs/hyperbolic_colors.dot")
print("Completed")


nx.draw(G,pos=posDict)  # networkx draw()
plt.show()
#plt.draw()  # pyplot draw()
#plt.show()
