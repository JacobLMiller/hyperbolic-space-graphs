import networkx as nx
import igraph as ig

import math
import random

import drawSvg as draw
from drawSvg import Drawing
from hyperbolic import euclid, util
from hyperbolic.poincare.shapes import *
from hyperbolic.poincare import Transform

def add_Edge(G,e1,e2):
    pass

def f_z0(z):
    pass

g = nx.drawing.nx_agraph.read_dot('test.dot')
G = {}
for v in list(g.nodes()):
    G[v] = {'euclidCoord': None,'neighbors': [],'Point': None}
    while True:
        z = (random.uniform(-1,1),random.uniform(-1,1))
        if not z[0]**2 + z[1]**2 >= 1:
            break
    G[v]['euclidCoord'] = z
    G[v]['Point'] = Point.fromEuclid(z[0],z[1])

paths = []

for e in list(g.edges()):
    print(e)
    add_Edge(G,e[0],e[1])
    paths.append(Line.fromPoints(*G[e[0]]['Point'],*G[e[1]]['Point'],segment=True))

#Begin drawing

d = Drawing(2.1, 2.1, origin='center')
d.draw(euclid.shapes.Circle(0, 0, 1), fill='#ddd')

for v in list(G.keys()):
    d.draw(G[v]['Point'], hradius=0.1, fill='green')
for e in paths:
    d.draw(e, hradius=0.01, fill='white')

d.setRenderSize(w=400)
d.saveSvg('secondGraph.svg')
print(G['0']['Point'].distanceTo(G['1']['Point']))
