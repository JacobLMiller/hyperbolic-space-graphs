import scipy.io
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import graph_tool.all as gt

H = nx.balanced_tree(2,8)
G = gt.Graph(directed=False)
G.add_vertex(n=len(H.nodes()))
for e in H.edges():
    G.add_edge(e[0],e[1])

G.save('btree8.dot')
