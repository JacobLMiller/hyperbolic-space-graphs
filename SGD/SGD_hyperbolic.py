import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import networkx as nx
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
#import tensorflow as tf
import drawSvg as draw
from math import sqrt
from SGD_MDS import myMDS


import math
import random
import cmath
import copy
import time
import os
import itertools

class HMDS:
    def __init__(self,dissimilarities,epsilon=0.1,init_pos=np.array([])):
        self.d = dissimilarities
        self.d_max = np.max(self.d)
        self.d_min = 1
        self.n = len(self.d)
        if init_pos.any(): #If an initial configuration is desired, assign it
            self.X = np.asarray(init_pos)
            if self.X.shape[0] != self.n:
                raise Exception("Number of elements in starting configuration must be equal to the number of elements in the dissimilarity matrix.")
        else: #Random point in the chosen geometry
            self.X = [[0,0] for i in range(self.n)]
            for i in range(len(self.d)):
                self.X[i] = self.init_point()
            self.X = np.asarray(self.X,dtype="float64")

        #Weight is inversely proportional to the square of the theoretic distance
        self.w = [[ 1/pow(self.d[i][j],2) if self.d[i][j] > 0 else 0 for i in range(self.n)]
                    for j in range(self.n)]

        #Values for step size calculation
        w_min = 1/pow(self.d_max,2)
        self.w_max = 1/pow(self.d_min,2)
        #epsilon = 0.1

        self.eta_max = 1/w_min
        self.eta_min = epsilon/self.w_max




    def solve(self,num_iter=1000,epsilon=1e-3,debug=False,schedule="default"):
        step,count  = self.eta_max, 0
        step = 1

        #Array of indices to shuffle and choose pairs of nodes
        #indices = [i for i in range(self.n)]
        indices = list(itertools.combinations(range(self.n), 2))
        random.shuffle(indices)

        weight = 1/choose(self.n,2)

        grads = np.zeros(self.X.shape)
        X = np.asarray(self.X)
        deltaX = 100
        max_change = 1
        uncapped = False

        while count < num_iter:# and max_change > 0.003:
            max_change = -100
            for k in range(len(indices)):
                i = indices[k][0]
                j = indices[k][1]
                if i > j:
                    i,j = j,i

                dist = polar_dist(X[i],X[j])
                dloss = grad(X[i],X[j])*((dist-self.d[i][j])/2)
                wc = step*self.w[i][j]
                if wc > 1:
                    wc = 1
                else:
                    uncapped = True

                m = (dloss*wc)
                X[i] = X[i] - m[0]
                X[j] = X[j] - m[1]

                new_dist = polar_dist(X[i],X[j])
                deltaX = abs(dist-new_dist)
                max_change = max(max_change,deltaX)


            step = self.compute_step_size_old(count,num_iter)

            #step = self.compute_step_size(count,num_iter,uncapped)


            count += 1
            #step = 1/count
            random.shuffle(indices)
            if debug:
                #self.X = X
                print(self.calc_stress())
                #print(step)

        self.X = X

        return self.X

    def calc_stress(self):
        """
        Calculates the standard measure of stress: \sum_{i,j} w_{i,j}(dist(Xi,Xj)-D_{i,j})^2
        Or, in English, the square of the difference of the realized distance and the theoretical distance,
        weighted by the table w, and summed over all pairs.
        """
        stress = 0
        for i in range(self.n):
            for j in range(i):
                stress += self.w[i][j]*pow(geodesic(self.X[i],self.X[j])-self.d[i][j],2)
        return stress

    def calc_distortion(self):
        """
        A normalized goodness of fit measure.
        """
        distortion = 0
        for i in range(self.n):
            for j in range(i):
                distortion += abs((geodesic(self.X[i],self.X[j])-self.d[i][j]))/self.d[i][j]
        return (1/choose(self.n,2))*distortion


    def compute_step_size_old(self,count,num_iter):
        lamb = math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return self.eta_max*math.exp(lamb*count)

    def compute_step_size(self,count,num_iter,uncapped):
        lamb = math.log(self.eta_min/self.eta_max)/30
        if uncapped:
            #return self.eta_max*math.exp(lamb*count)
            return self.w_max/pow(1+5*count,0.5)
        else:
            return self.eta_max*math.exp(lamb*count)


    def init_point(self):
            r = pow(random.uniform(0,1),0.5)
            theta = random.uniform(0,2*math.pi)
            x = math.atanh(math.tanh(r)*math.cos(theta))
            y = math.asinh(math.sinh(r)*math.sin(theta))
            return [r,theta]

def grad(p,q):
    r,t = p
    a,b = q
    sin = np.sin
    cos = np.cos
    sinh = np.sinh
    cosh = np.cosh
    bottom = 1/pow(pow(part_of_dist(p,q),2)-1,0.5)

    delta_a = -(cos(b-t)*sinh(r)*cosh(a)-sinh(a)*cosh(r))*bottom
    delta_b = (sin(b-t)*sinh(a)*sinh(r))*bottom

    delta_r = -1*(cos(b-t)*sinh(a)*cosh(r)-sinh(r)*cosh(a))*bottom
    delta_t = -1*(sin(b-t)*sinh(a)*sinh(r))*bottom

    return np.array([[delta_r,delta_t],[delta_a,delta_b]])

def part_of_dist(xi,xj):
    r,t = xi
    a,b = xj
    sinh = np.sinh
    cosh = np.cosh
    #print(np.cosh(y1)*np.cosh(x2-x1)*np.cosh(y2)-np.sinh(y1)*np.sinh(y2))
    return np.cos(b-t)*sinh(a)*sinh(r)-cosh(a)*cosh(r)


def normalize(v):
    mag = pow(v[0]*v[0]+v[1]*v[1],0.5)
    return v/mag

def polar_dist(x1,x2):
    r1,theta1 = x1
    r2,theta2 = x2
    return np.arccosh(np.cosh(r1)*np.cosh(r2)-np.sinh(r1)*np.sinh(r2)*np.cos(theta2-theta1))

def geodesic(xi,xj):
    return polar_dist(xi,xj)

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
            myList.append(-1)

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
    nx.draw(G,pos=pos,with_labels=True)
    plt.show()
    plt.clf()

def output_hyperbolic(X,G):
    count = 0
    for i in G.nodes():
        #print(X)
        x,y = X[count]
        Rh = float(x)
        theta = float(y)
        #Rh = np.arccosh(np.cosh(x)*np.cosh(y))
        #theta = 2*math.atan2(np.sinh(x)*np.cosh(y)+pow(pow(np.cosh(x),2)*pow(np.cosh(y),2)-1,0.5),np.sinh(y))
        #Re = (math.exp(Rh)-1)/(math.exp(Rh)+1)
        #hR = math.acosh((r*r/2)+1)
        Rl = pow(2*(math.cosh(Rh)-1),0.5)

        G.nodes[i]['mypos'] = str(Rh) + "," + str(theta)
        #G.nodes[i]['pos'] = str(500*Rl*math.cos(theta)) + "," + str(500*Rl*math.sin(theta))

        count += 1
    nx.drawing.nx_agraph.write_dot(G, "output_hyperbolic.dot")
    nx.drawing.nx_agraph.write_dot(G, "/home/jacob/Desktop/hyperbolic-space-graphs/old/jsCanvas/graphs/hyperbolic_colors.dot")
    nx.drawing.nx_agraph.write_dot(G, "/home/jacob/Desktop/hyperbolic-space-graphs/maps/static/graphs/hyperbolic_colors.dot")

#Program start
def main():
    #G = nx.drawing.nx_agraph.read_dot('input.dot')
    G = nx.triangular_lattice_graph(5,5)
    #G = nx.full_rary_tree(2,30)
    d = np.asarray(all_pairs_shortest_path(G))/1
    d = d*1

    best_X = []
    best_score = 1000000
    Z = myMDS(d)
    Z.solve(2)
    init = np.ones(Z.X.shape)
    for i in range(len(Z.X)):
        r = pow(pow(Z.X[i][0],2)+pow(Z.X[i][1],2),0.5)
        theta = math.atan2(Z.X[i][1],Z.X[i][0])
        init[i] = np.array([r,theta])
    #print(Z.X)
    #print(Z.calc_stress())
    #output_euclidean(G,Z.X)
    for i in range(1):
        Y = HMDS(d,epsilon=0.25)
        Y.solve(15,debug=True)
        print(Y.calc_stress())
        if Y.calc_stress() < best_score:
            best_score = Y.calc_stress()
            print(Y.calc_distortion())
            best_X = Y.X
            print('got better')
        #print(i)
    output_hyperbolic(best_X,G)
    #print(best_score)
    #output_euclidean(best_X)
#main()
