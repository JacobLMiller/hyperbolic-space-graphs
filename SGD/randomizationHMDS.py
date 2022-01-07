import networkx as nx
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
#import tensorflow as tf
import drawSvg as draw
from math import sqrt
import itertools
from SGD_MDS import myMDS

from drawSvg import Drawing
from hyperbolic import euclid, util
from hyperbolic.poincare.shapes import *
from hyperbolic.poincare import Transform

import math
import random
import cmath
import copy
import time
import os
#from SGD_MDS import MDS

class HMDS:
    def __init__(self,dissimilarities,init_pos=np.array([])):
        self.d = dissimilarities
        self.d_max = np.max(dissimilarities)
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
        self.w = [[1/pow(self.d[i][j],3.14) if self.d[i][j] > 0 else 0 for i in range(self.n)]
                    for j in range(self.n)]

        #Values for step size calculation
        w_min = 1/pow(self.d_max,2)
        w_max = 1/pow(self.d_min,2)
        epsilon = 0.1

        self.eta_max = 1/w_min
        self.eta_min = epsilon/w_max




    def solve_rr(self,num_iter=1000,epsilon=1e-3,debug=False):
        step,count  = self.eta_max, 0

        #Array of indices to shuffle and choose pairs of nodes
        indices = list(itertools.combinations(range(self.n), 2))
        random.shuffle(indices)
        weight = 1/choose(self.n,2)

        double_count = 0
        loss = []

        while count < num_iter:
            for k in range(len(indices)):
                i = indices[k][0]
                j = indices[k][1]
                if i > j:
                    i,j = j,i

                pq = self.X[i] - self.X[j] #Vector between points
                #pq = grad(self.X[i],self.X[j])
                #print(grad(self.X[i],self.X[j]))
                mag = geodesic(self.X[i],self.X[j])

                r = (mag-self.d[i][j]) #min distance each node needs to move to satisfy d[i][j]

                wc = self.w[i][j]*step #Weighted step size. If > 1, set it to 1
                if wc > 1:
                    wc = 1
                r = wc*r

                m = pq*r / mag

                #self.X[i] = self.X[i] - m
                #self.X[j] = self.X[j] + m

                #mag = geodesic(self.X[i],self.X[j])
                                #print(dist_grad)
                if True:
                    #print(dist_grad)
                    dist_grad = (self.X[i]-self.X[j])

                    #print(dist_grad)


                    r = (mag-self.d[i][j])/2

                    wc = self.w[i][j]*step
                    if wc > 1:
                        wc = 1
                    r = wc*r

                    m = dist_grad*r/mag


                    self.X[i] = self.X[i] - m
                    self.X[j] = self.X[j] + m
                elif False:
                    dist_grad = grad_old(self.X[i],self.X[j])
                    T = weight*2*self.w[i][j]*dist_grad*(mag-self.d[i][j])
                    self.X[i] = self.X[i] - step*T[0]
                    self.X[j] = self.X[j] - step*T[1]

            count += 1

            if debug:
                stress = self.calc_stress()
                #print(stress)
                loss.append(stress)

            step = self.compute_step_size(count,num_iter)

            #count += 1
            #output_hyperbolic(self.X,G,count)
            random.shuffle(indices)


        return loss

    def solve_sample_replacement(self,num_iter=1000,epsilon=1e-3,debug=False):
        step,count  = self.eta_max, 0

        #Array of indices to shuffle and choose pairs of nodes
        indices = [i for i in range(self.n)]
        #indices = list(itertools.combinations(range(self.n), 2))
        random.shuffle(indices)
        weight = 1/choose(self.n,2)

        double_count = 0
        loss = []

        while count < num_iter:
            i = indices[random.randrange(0,self.n)]
            j = indices[random.randrange(0,self.n)]
            while i == j:
                j = indices[random.randrange(0,self.n)]
            if i > j:
                i,j = j,i

            pq = self.X[i] - self.X[j] #Vector between points
            #pq = grad(self.X[i],self.X[j])
            #print(grad(self.X[i],self.X[j]))
            mag = geodesic(self.X[i],self.X[j])

            r = (mag-self.d[i][j]) #min distance each node needs to move to satisfy d[i][j]

            wc = self.w[i][j]*step #Weighted step size. If > 1, set it to 1
            if wc > 1:
                wc = 1
            r = wc*r

            m = pq*r / mag

            #self.X[i] = self.X[i] - m
            #self.X[j] = self.X[j] + m

            #mag = geodesic(self.X[i],self.X[j])
                            #print(dist_grad)
            if True:
                #print(dist_grad)
                dist_grad = (self.X[i]-self.X[j])

                #print(dist_grad)


                r = (mag-self.d[i][j])/2

                wc = self.w[i][j]*step
                if wc > 1:
                    wc = 1
                r = wc*r

                m = dist_grad*r/mag


                self.X[i] = self.X[i] - m
                self.X[j] = self.X[j] + m
            elif False:
                dist_grad = grad_old(self.X[i],self.X[j])
                T = weight*2*self.w[i][j]*dist_grad*(mag-self.d[i][j])
                self.X[i] = self.X[i] - step*T[0]
                self.X[j] = self.X[j] - step*T[1]

            if count % 10 == 0:
                #Draw_SVG(self.X,double_count)
                double_count += 1
            if debug:
                stress = self.calc_stress()
                #print(stress)
                loss.append(stress)
                if count >= num_iter:
                    break

            if count % self.n == 0:
                step = self.compute_step_size(count,num_iter)

            count += 1
            #output_hyperbolic(self.X,G,count)
            random.shuffle(indices)

        return loss

    def solve_shuffle_indices(self,num_iter=1000,epsilon=1e-3,debug=False):
        step,count  = self.eta_max, 0

        #Array of indices to shuffle and choose pairs of nodes
        indices = [i for i in range(self.n)]
        #indices = list(itertools.combinations(range(self.n), 2))
        random.shuffle(indices)
        weight = 1/choose(self.n,2)

        double_count = 0
        loss = []

        while count < num_iter:
            for k in range(0,self.n-1,2):
                i = indices[k]
                j = indices[k+1]

                if i > j:
                    i,j = j,i

                pq = self.X[i] - self.X[j] #Vector between points
                #pq = grad(self.X[i],self.X[j])
                #print(grad(self.X[i],self.X[j]))
                mag = geodesic(self.X[i],self.X[j])

                r = (mag-self.d[i][j]) #min distance each node needs to move to satisfy d[i][j]

                wc = self.w[i][j]*step #Weighted step size. If > 1, set it to 1
                if wc > 1:
                    wc = 1
                r = wc*r

                m = pq*r / mag

                #self.X[i] = self.X[i] - m
                #self.X[j] = self.X[j] + m

                #mag = geodesic(self.X[i],self.X[j])
                                #print(dist_grad)
                if True:
                    #print(dist_grad)
                    dist_grad = (self.X[i]-self.X[j])

                    #print(dist_grad)


                    r = (mag-self.d[i][j])/2

                    wc = self.w[i][j]*step
                    if wc > 1:
                        wc = 1
                    r = wc*r

                    m = dist_grad*r/mag


                    self.X[i] = self.X[i] - m
                    self.X[j] = self.X[j] + m
                elif False:
                    dist_grad = grad_old(self.X[i],self.X[j])
                    T = weight*2*self.w[i][j]*dist_grad*(mag-self.d[i][j])
                    self.X[i] = self.X[i] - step*T[0]
                    self.X[j] = self.X[j] - step*T[1]

                count += 1

                if debug:
                    stress = self.calc_stress()
                    #print(stress)
                    loss.append(stress)
                    if count >= num_iter:
                        break
                if count % self.n == 0:
                    step = self.compute_step_size(count,num_iter)
            random.shuffle(indices)


        return loss

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
        return stress/2

    def calc_distortion(self):
        """
        A normalized goodness of fit measure.
        """
        distortion = 0
        for i in range(self.n):
            for j in range(i):
                distortion += abs((geodesic(self.X[i],self.X[j])-self.d[i][j]))/self.d[i][j]
        return (1/choose(self.n,2))*distortion


    def compute_step_size(self,count,num_iter):
        lamb = math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return self.eta_max*math.exp(lamb*count)


    def init_point(self):
            r = pow(random.uniform(0,1),0.5)
            theta = random.uniform(0,2*math.pi)
            x = math.atanh(math.tanh(r)*math.cos(theta))
            y = math.asinh(math.sinh(r)*math.sin(theta))
            return [x,y]


def normalize(v):
    mag = pow(sum([val*val for val in v]), 0.5)
    return np.array([val/mag for val in v])

def mobius(z,a,b,c,d):
    return a*z+b/c*z+d

def geodesic(xi,xj):
    return lob_dist(xi,xj)

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product


def grad(p,q):
    r,t = p
    a,b = q
    sin = np.sin
    cos = np.cos
    sinh = math.sinh
    cosh = math.cosh
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

def grad_old(p,q):
    x,y = p
    a,b = q
    sinh = math.sinh
    cosh = math.cosh
    bottom = 1/pow(pow(part_of_dist_old(p,q),2)-1,0.5)

    delta_a = sinh(a-x)*cosh(b)*cosh(y)*bottom
    delta_b = -1*(-sinh(b)*cosh(y)*cosh(a-x)+sinh(y)*cosh(b))*bottom

    delta_x = -1*sinh(a-x)*cosh(b)*cosh(y)*bottom
    delta_y = (-sinh(b)*cosh(y) + sinh(y)*cosh(b)*cosh(a-x))*bottom

    return np.array([[delta_x,delta_y],[delta_a,delta_b]])

def part_of_dist_old(xi,xj):
    x,y = xi
    a,b = xj
    #print(np.cosh(y1)*np.cosh(x2-x1)*np.cosh(y2)-np.sinh(y1)*np.sinh(y2))
    return np.sinh(b)*np.sinh(y)-np.cosh(b)*np.cosh(y)*np.cosh(a-x)


def polar_dist(x1,x2):
    r1,theta1 = x1
    r2,theta2 = x2
    return np.arccosh(np.cosh(r1)*np.cosh(r2)-np.sinh(r1)*np.sinh(r2)*np.cos(theta2-theta1))



def lob_dist(xi,xj):
    x1,y1 = xi
    x2,y2 = xj
    dist = np.arccosh(np.cosh(y1)*np.cosh(x2-x1)*np.cosh(y2)-np.sinh(y1)*np.sinh(y2))
    if np.isnan(dist):
        return 200
    return dist


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


def output_euclidean(X):
    pos = {}
    count = 0
    for x in G.nodes():
        pos[x] = X[count]
        count += 1
    nx.draw(G,pos=pos,with_labels=True)
    plt.show()
    plt.clf()

def Draw_SVG(X,number):
    points = []
    lines = []
    nodeDict = {}
    d = Drawing(2.1,2.1, origin='center')
    d.draw(euclid.shapes.Circle(0, 0, 1), fill='#ddd')

    count = 0
    for i in G.nodes():
        x,y= X[count]
        Rh = np.arccosh(np.cosh(x)*np.cosh(y))
        theta = 2*math.atan2(np.sinh(x)*np.cosh(y)+pow(pow(np.cosh(x),2)*pow(np.cosh(y),2)-1,0.5),np.sinh(y))
        Re = (math.exp(Rh)-1)/(math.exp(Rh)+1)
        #Re = (math.exp(r)-1)/(math.exp(r)+1)
        x = Re*np.cos(theta)
        y = Re*np.sin(theta)
        G.nodes[i]['pos'] = [x,y]
        count += 1

    for i in G.nodes:
        #print(G.nodes[i]['pos'])
        #print(cmath.polar(complex(*G.nodes[i]['pos'])))
        points.append(Point(G.nodes[i]['pos'][0],G.nodes[i]['pos'][1]))
        nodeDict[i] = points[-1]

    for i in G.edges:
        lines.append(Line.fromPoints(*nodeDict[i[0]],*nodeDict[i[1]],segment=True))

    #trans = Transform.shiftOrigin(points[0])

    for i in lines:
        d.draw(i,hwidth=.01,fill='black')

    for i in points:
        d.draw(i,hradius=.05,fill='green')


    d.setRenderSize(w=1000)
    d.saveSvg('SGD/slideshow/Test' + str(number) + '.svg')

def output_hyperbolic(X,G,count1):
    count = 0
    for i in G.nodes():
        x,y = X[count]
        Rh = x
        theta = y
        Rh = np.arccosh(np.cosh(x)*np.cosh(y))
        theta = 2*math.atan2(np.sinh(x)*np.cosh(y)+pow(pow(np.cosh(x),2)*pow(np.cosh(y),2)-1,0.5),np.sinh(y))
        Re = (math.exp(Rh)-1)/(math.exp(Rh)+1)
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
    #G = nx.triangular_lattice_graph(5,5)
    #G = nx.random_tree(50)

    d = np.asarray(all_pairs_shortest_path(G))

    best_X = []
    best_score = 10000000

    for i in range(1):
        Y = HMDS(d)
        Y.solve(15)
        if Y.calc_stress() < best_score:
            best_score = Y.calc_distortion()
            best_X = Y.X
            print('got better')
        print(i)
    output_hyperbolic(best_X,G,0)
    print(best_score)
#g = ig.Graph.Tree(500,2)
#g.write_dot('input.dot')

#G = nx.drawing.nx_agraph.read_dot('input.dot')
#d = np.array(all_pairs_shortest_path(G))/1
#main()
