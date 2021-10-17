import networkx as nx
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
#import tensorflow as tf
import drawSvg as draw
from math import sqrt
import sys
import itertools


import math
import random
import cmath
import copy
import time
import os

class myMDS:
    def __init__(self,dissimilarities,init_pos=np.array([])):
        self.d = dissimilarities
        self.d_max = np.max(dissimilarities)
        self.d_min = 1
        self.n = len(self.d)
        if init_pos.any():
            self.X = np.asarray(init_pos)
        else: #Random point in the chosen geometry
            self.X = np.zeros((len(self.d),2))
            for i in range(len(self.d)):
                self.X[i] = self.init_point()
            self.X = np.asarray(self.X)

        self.w = [[ 1/pow(self.d[i][j],2) if i != j else 0 for i in range(self.n)]
                    for j in range(self.n)]
        for i in range(len(self.d)):
            self.w[i][i] = 0

        w_min = 1/pow(self.d_max,2)
        w_max = 1/pow(self.d_min,2)
        self.eta_max = 1/w_min
        epsilon = 0.1
        self.eta_min = epsilon/w_max




    def solve(self,num_iter=1000,epsilon=1e-3,debug=False):
        current_error,delta_e,step,count = 1000,1,self.eta_max,0
        #indices = [i for i in range(len(self.d))]
        indices = list(itertools.combinations(range(self.n), 2))
        random.shuffle(indices)
        #random.shuffle(indices)

        weight = 1/choose(self.n,2)

        while count < num_iter:
            for k in range(len(indices)):
                i = indices[k][0]
                j = indices[k][1]
                if i > j:
                    i,j = j,i

                pq = self.X[i] - self.X[j] #Vector between points



                #mag1 = math.sqrt(pq[0]*pq[0]+pq[1]*pq[1]) #Magnitude |p-q|
                mag = geodesic(self.X[i],self.X[j])
                #if abs(mag-mag1) > 1e-2:
                #    print('somethings not right')
                r = (mag-self.d[i][j])/2 #min distance to satisfy constraint

                wc = self.w[i][j]*step
                if wc > 1:
                    wc = 1
                r = wc*r

                term3 = (mag-self.d[i][j])/2
                #self.X[i] = self.X[i] - (wc*pq*term3)/geodesic(self.X[i],self.X[j])
                #self.X[j] = self.X[j] + wc*pq*term3/geodesic(self.X[i],self.X[j])

                m = (pq*term3*wc)/mag

                self.X[i] = self.X[i] - m
                self.X[j] = self.X[j] + m

            step = self.compute_step_size(count,num_iter)


            count += 1
            random.shuffle(indices)
            if debug:
                print(self.calc_stress())

        return self.X

    def calc_stress(self):
        stress = 0
        for i in range(self.n):
            for j in range(i):
                stress += self.w[i][j]*pow(geodesic(self.X[i],self.X[j])-self.d[i][j],2)
        return stress

    def calc_distortion(self):
        distortion = 0
        for i in range(self.n):
            for j in range(i):
                distortion += abs((geodesic(self.X[i],self.X[j])-self.d[i][j]))/self.d[i][j]
        return (1/choose(self.n,2))*distortion

    def calc_gradient(self,i,j):
        X0 = tf.Variable(self.X)
        with tf.GradientTape() as tape:
            Y = self.calc_stress(X0)
        dy_dx = tape.gradient(Y,X0).numpy()
        #dy = dy_dx.numpy()
        for i in range(len(self.d)):
            dy_dx[i] = normalize(dy_dx[i])
        return dy_dx

    def compute_step_size(self,count,num_iter):
        lamb = math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return self.eta_max*math.exp(lamb*count)


    def init_point(self):
        return [random.uniform(-1,1),random.uniform(-1,1)]


def normalize(v):
    mag = pow(v[0]*v[0]+v[1]*v[1],0.5)
    return v/mag

def geodesic(xi,xj):
    return euclid_dist(xi,xj)

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product


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
        if x in distance:
            myList.append(distance[x])
        else:
            myList.append(len(G.nodes)+1)

    return myList

def all_pairs_shortest_path(G):
    d = [ [ -1 for i in range(len(G.nodes)) ] for j in range(len(G.nodes)) ]

    count = 0
    for node in G.nodes:
        d[count] = bfs(G,node)
        count += 1
    return d

def scale_matrix(d,new_max):
    d_new = np.zeros(d.shape)

    t_min,r_min,r_max,t_max = 0,0,np.max(d),new_max

    for i in range(d.shape[0]):
        for j in range(i):
            m = ((d[i][j]-r_min)/(r_max-r_min))*(t_max-t_min)+t_min
            d_new[i][j] = m
            d_new[j][i] = m
    return d_new


def euclid_dist(x1,x2):
    x = x2[0]-x1[0]
    y = x2[1]-x1[1]
    return pow(x*x+y*y,0.5)


def output_euclidean(G,X):
    pos = {}
    count = 0
    for x in G.nodes():
        pos[x] = X[count]
        count += 1
    nx.draw(G,pos=pos)
    plt.show()
    plt.clf()

    count = 0
    for i in G.nodes():
        x,y = X[count]
        G.nodes[i]['pos'] = str(100*x) + "," + str(100*y)

        count += 1
    nx.drawing.nx_agraph.write_dot(G, "output.dot")

#Code start

G = nx.triangular_lattice_graph(5,5)
#G = nx.drawing.nx_agraph.read_dot('input.dot')
#G = nx.full_rary_tree(2,100)
d = np.array(all_pairs_shortest_path(G))/1

Y = myMDS(d)
#Y.solve(15,debug=False)
#print(Y.calc_distortion())
#output_euclidean(G,Y.X)
