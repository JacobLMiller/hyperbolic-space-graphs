import networkx as nx
import igraph as ig

import math
import random

import drawSvg as draw
from drawSvg import Drawing
from hyperbolic import euclid, util
from hyperbolic.poincare.shapes import *
from hyperbolic.poincare import Transform

g2 = ig.Graph.Famous("Coxeter")

ig.Graph.write_dot(g2,'Coxeter.dot')
