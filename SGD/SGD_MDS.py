import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
#import tensorflow as tf

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
    def __init__(self,dissimilarities,k=5,weighted=False,init_pos=np.array([])):
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

        a = 1
        b = 1
        if weighted:
            self.w = set_w(self.d,k)
            print(self.w)
        else:
            self.w = [[ 1/math.exp(self.d[i][j]-1) if i != j else 0 for i in range(self.n)]
                        for j in range(self.n)]

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

        weight = 1

        while count < num_iter:
            for k in range(len(indices)):
                i = indices[k][0]
                j = indices[k][1]
                if i > j:
                    i,j = j,i

                pq = self.X[i] - self.X[j] #Vector between points

                mag = geodesic(self.X[i],self.X[j])
                r = (mag-self.d[i][j])/2 #min distance to satisfy constraint

                wc = self.w[i][j]*step
                if wc > 1:
                    wc = 1
                r = wc*r

                m = (pq*r)/mag

                self.X[i] = self.X[i] - m
                self.X[j] = self.X[j] + m

                #save_euclidean(self.X,weight)
                #weight += 1

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

def save_euclidean(X,number):
    pos = {}
    count = 0
    for i in G.nodes():
        x,y = X[count]
        pos[i] = [x,y]
        count += 1
    nx.draw(G,pos=pos,with_labels=True)
    plt.savefig('test'+str(number)+'.png')
    plt.clf()

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

def set_w(d,k):
    k_nearest = [get_k_nearest(d[i],k) for i in range(len(d))]

    #1/(10*math.exp(d[i][j]))
    w = np.asarray([[ 0.001 if i != j else 0 for i in range(len(d))] for j in range(len(d))])
    for i in range(len(d)):
        for j in k_nearest[i]:
            if i != j:
                w[i][j] = 1


    return w
def get_k_nearest(d_row,k):
    return np.argpartition(d_row,k)[:k+1]

#Code start

#G = nx.grid_graph([10,5])
# G = nx.random_partition_graph([30,10,40,5], 0.8, 0.01)
# #print(G.nodes[40])
# #G = nx.drawing.nx_agraph.read_dot('input.dot')
# #G = nx.full_rary_tree(2,100)
# #G = nx.random_tree(500)
# #g = ig.Graph.Tree(500,2)
# #g.write_dot('input.dot')
# #G = generate_graph(100,0.5)
#
# #G = nx.drawing.nx_agraph.read_dot('input.dot')
# d = np.array(all_pairs_shortest_path(G))/1
#
# Y = myMDS(d,weighted=True,k=5)
# Y.solve(30,debug=False)
#
# output_euclidean(G,Y.X)
