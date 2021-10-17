import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import random
import tensorflow as tf
import numpy as np
import networkx as nx
#from SGD_MDS import all_pairs_shortest_path

from SGD_hyperbolic import HMDS, all_pairs_shortest_path
from MDS import MDS

def lambert_azimuthal(p1):
    r,theta = p1
    newR = pow(2*(math.cosh(r)-1),0.5)
    return (newR,theta)

def inverse_lamb_azimuthal(p1):
    r,theta = p1
    newR = math.acosh((r*r + 2)/2)
    return (newR,theta)

def lob_dist(xi,xj):
    dist = tf.math.acosh(tf.math.cosh(xi[1])*tf.math.cosh(xj[0]-xi[0])*tf.math.cosh(xj[1])-tf.math.sinh(xi[1])*tf.math.sinh(xj[1]))
    return dist

def lob_dist_2(xi,xj):
    dist = np.arccosh(np.cosh(xi[1])*np.cosh(xj[0]-xi[0])*np.cosh(xj[1])-np.sinh(xi[1])*np.sinh(xj[1]))
    return dist

def polar_dist(x1,x2):
    return tf.math.acosh(tf.math.cosh(x1[0])*tf.math.cosh(x2[0])-tf.math.sinh(x1[0])*tf.math.sinh(x2[0])*tf.math.cos(x2[1]-x1[1]))

def polar_dist_2(x1,x2):
    r1,theta1 = x1
    r2,theta2 = x2
    return np.arccosh(np.cosh(r1)*np.cosh(r2)-np.sinh(r1)*np.sinh(r2)*np.cos(theta2-theta1))


def sphere_grad(p,q):
    lamb1,phi1 = p
    lamb2,phi2 = q
    bottom = 1/pow(1-pow(part_of_dist_sphere(p,q),2),0.5)
    sin = np.sin
    cos = np.cos
    x = -sin(lamb2-lamb1)*cos(phi2)*cos(phi1)*bottom
    y = (-sin(phi2)*cos(phi1) + sin(phi1)*cos(phi2)*cos(lamb2-lamb1))*bottom
    return np.array([x,y])

def part_of_dist_sphere(xi,xj):
    lamb1,phi1 = xi
    lamb2,phi2 = xj
    sin = np.sin
    cos = np.cos
    return sin(phi1)*sin(phi2) + cos(phi1)*cos(phi2)*cos(lamb2-lamb1)


def stress(xi,xj,d):
    dist = polar_dist(xi,xj)
    return (1/pow(d,2))*pow(dist-d,2)

def stress_grad(xi,xj,d):
    dist = polar_dist_2(xi,xj)
    return 2*(1/pow(d,2))*(dist-d)*grad(xi,xj)

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

def euclid_dist(x1,x2):

    return pow(pow(x1-x2,2).sum(),0.5)
def euclid_dist_2(x1,x2):
    return pow(tf.reduce_sum(pow(x1-x2,2)),0.5)

def sphere_dist(xi,xj):
    lamb1,phi1 = xi
    lamb2,phi2 = xj
    sin = np.sin
    cos = np.cos
    return np.arccos(sin(phi1)*sin(phi2) + cos(phi1)*cos(phi2)*cos(lamb2-lamb1))

def sphere_dist_2(xi,xj):
    lamb1,phi1 = xi
    lamb2,phi2 = xj
    sin = tf.math.sin
    cos = tf.math.cos
    return tf.math.acos(sin(phi1)*sin(phi2) + cos(phi1)*cos(phi2)*cos(lamb2-lamb1))

G = nx.triangular_lattice_graph(2,2)
d = np.array(all_pairs_shortest_path(G))/1

Y = MDS(d,geometry='spherical')

print(euclid_dist(np.array([0,0]),np.array([0,1])))

tfX = tf.Variable(Y.X)

with tf.GradientTape() as tape:
    tape.watch(tfX)

    loss = 0
    for i in range(len(Y.X)):
        for j in range(i):
            loss += pow(1/d[i][j],2)*pow(sphere_dist_2(tfX[i],tfX[j])-d[i][j],2)

# Auto-diff magic!  Calcs gradients between loss calc and params
dloss_dpos = tape.gradient(loss, tfX)
print(dloss_dpos)

X = np.asarray(Y.X)

loss = np.zeros(X.shape)
for i in range(len(Y.X)):
    for j in range(len(Y.X)):
        if i != j:
            dist = sphere_dist(X[i],X[j])
            loss[i] += 2*pow(1/d[i][j],2)*sphere_grad(X[i],X[j])*(dist-d[i][j])
print(loss)
