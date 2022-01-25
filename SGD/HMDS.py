import numpy as np
#import tensorflow as tf
import math
import random
import itertools


from numba import jit

import pygraphviz
import graph_tool.all as gt
import io

@jit(nopython=True)
def geodesic(u,v):
    x1,y1 = u
    x2,y2 = v
    dist = np.arccosh(np.cosh(y1)*np.cosh(x2-x1)*np.cosh(y2)-np.sinh(y1)*np.sinh(y2))
    if np.isnan(dist):
        dist = np.linalg.norm(u-v)
    return dist

@jit(nopython=True)
def satisfy(u,v,d,w,step):
    """
    u,v: hyperbolic vectors
    d: ideal distance between u and v from shortest path matrix
    w: associated weight of the pair u,v
    step: Fraction of distance u and v should be moved along gradient
    Returns: updated hyperbolic vectors of u and v

    Code modified from https://github.com/jxz12/s_gd2 and associated paper.
    """
    pq = u-v
    mag = geodesic(u,v)
    r = (mag-d)/2

    wc = w*step
    if wc > 1:
        wc = 1
    r = wc*r
    m = pq*r /mag

    return u-m, v+m

@jit(nopython=True)
def step_func1(count):
    return 1/(5+count)

@jit(nopython=True)
def calc_stress(X,d,w):
    """
    Calculates the standard measure of stress: \sum_{i,j} w_{i,j}(dist(Xi,Xj)-D_{i,j})^2
    Or, in English, the square of the difference of the realized distance and the theoretical distance,
    weighted by the table w, and summed over all pairs.
    """
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += w[i][j]*pow(geodesic(X[i],X[j])-d[i][j],2)
    return pow(stress,0.5)

@jit(nopython=True)
def choose1(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

@jit(nopython=True)
def calc_distortion(X,d,w):
    distortion = 0
    for i in range(len(X)):
        for j in range(i):
            distortion += abs((geodesic(X[i],X[j])-d[i][j]))/d[i][j]
    return (1/choose1(len(X),2))*distortion

@jit(nopython=True)
def stoch_solver(X,d,w,indices,schedule,num_iter=15,epsilon=1e-3):
    step = 0.1
    shuffle = random.shuffle

    for count in range(num_iter):

        for i,j in indices: # Random pair
            X[i],X[j] = satisfy(X[i],X[j],d[i][j],w[i][j],step) #Gradient w.r.t. pair i and j

        step = schedule[count] if count <= len(schedule) else schedule[-1] #Get next step size
        shuffle(indices) #Shuffle pair order
        if step > 0.1:
            step = 0.1



    return X

@jit(nopython=True)
def stoch_solver_debug(X,d,w,indices,schedule,num_iter=15,epsilon=1e-3):
    step = 1
    shuffle = random.shuffle
    print(schedule)
    yield X.copy()
    for count in range(num_iter):
        for i,j in indices: # Random pair
            X[i],X[j] = satisfy(X[i],X[j],d[i][j],w[i][j],step) #Gradient w.r.t. pair i and j

        step = schedule[count] if count <= len(schedule) else schedule[-1]/10 #Get next step size

        shuffle(indices) #Shuffle pair order
        print(calc_stress(X,d,w))
        yield X.copy()

    return X

@jit(nopython=True)
def set_step(w_max,eta_max,eta_min):
    a = 1/w_max
    b = -np.log(eta_min/eta_max)/(15-1)
    step = lambda count: eta_max*np.exp(-b*count)

    # lamb = np.log(eta_min/eta_max)/(15-1)
    # step = lambda count: np.exp(lamb*count)

    return np.array([step(count) for count in range(15)])

def preprocess(graph,input_format='dot'):
    if True:
        graph_file = io.StringIO(pygraphviz.AGraph(graph).to_string())
        G = gt.load_graph(graph_file,fmt='dot')
        print("yo")
        return G,get_distance_matrix(G)

def postprocess(G,embedding):
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(embedding.T)
    G.vertex_properties['pos'] = pos

    import tempfile
    with tempfile.TemporaryFile() as file:
        G.save(file,fmt='dot')
        file.seek(0)
        dot_rep = file.read()
    return gt_to_json(G,embedding), dot_rep

def gt_to_json(G,embedding):
    nodes, edges = G.iter_vertices(),G.iter_edges()

    out = {"nodes": [None for i in range(G.num_vertices())],
            "edges": [None for i in range(G.num_edges())]
            }
    for v in nodes:
        out["nodes"][int(v)] = {
            "id": int(v),
            "pos": list(embedding[int(v)])
        }
    count = 0
    for u,v in edges:
        ##Implement map or zip or something
        out["edges"][count] = {
            "s": int(u),
            "t": int(v)
        }
        count += 1

    return out

class HMDS:
    def __init__(self,dissimilarities,init_pos=np.empty(1)):
        self.d = dissimilarities
        self.d_max = np.max(dissimilarities)
        #self.d = self.d*(2*math.pi/self.d_max)
        self.d_min = 1
        self.n = len(self.d)
        if self.n > 30:
            self.d = self.d*(10/self.d_max)
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
        self.w = np.array([[1/pow(self.d[i][j],2) if self.d[i][j] > 0 else 0 for i in range(self.n)]
                            for j in range(self.n)])

        #Values for step size calculation
        w_min = 1/pow(self.d_max,2)
        self.w_max = 1/pow(self.d_min,2)

        self.eta_max = 1/w_min
        self.eta_min = 0.1/self.w_max

        self.indices = np.array(list(itertools.combinations(range(self.n), 2)))

        self.steps = set_step(self.w_max,self.eta_max,self.eta_min)


    def solve(self,num_iter=20,debug=False):
        X = self.X
        d = self.d
        w = self.w
        if debug:
            solve_step = stoch_solver_debug(X,d,w,self.indices,self.steps,num_iter=num_iter)
            #print(next(solve_step))
            Xs = [x for x in solve_step]
            self.stress_hist = [calc_stress(x,d,w) for x in Xs]
            self.X =  Xs[-1]
            return

        X = stoch_solver(self.X,self.d,self.w,self.indices,self.steps,num_iter=num_iter)
        self.X = X

    def calc_stress3(self):
        """
        Calculates the standard measure of stress: \sum_{i,j} w_{i,j}(dist(Xi,Xj)-D_{i,j})^2
        Or, in English, the square of the difference of the realized distance and the theoretical distance,
        weighted by the table w, and summed over all pairs.
        """
        stress = 0
        for i in range(self.n):
            for j in range(i):
                stress += self.w[i][j]*pow(geodesic(self.X[i],self.X[j])-self.d[i][j],2)
        return pow(stress,0.5)

    def calc_stress2(self):
        """
        Calculates the standard measure of stress: \sum_{i,j} w_{i,j}(dist(Xi,Xj)-D_{i,j})^2
        Or, in English, the square of the difference of the realized distance and the theoretical distance,
        weighted by the table w, and summed over all pairs.
        """
        stress = 0
        for i in range(self.n):
            for j in range(i):
                stress += self.w[i][j]*pow(geodesic(self.X[i],self.X[j])-self.d[i][j],2)
        bottom = 0
        for i in range(self.n):
            for j in range(i):
                bottom += self.d[i][j] ** 2
        return stress/bottom

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

    def compute_step_size(self,count,num_iter):
        a = 1/self.w_max
        b = -math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return a/(pow(1+b*count,0.5))

    def init_point(self):
            r = pow(random.uniform(0,1),0.5)
            theta = random.uniform(0,2*math.pi)
            x = math.atanh(math.tanh(r)*math.cos(theta))
            y = math.asinh(math.sinh(r)*math.sin(theta))
            return [x,y]


def normalize(v):
    mag = pow(sum([val*val for val in v]), 0.5)
    return np.array([val/mag for val in v])



def lob_dist(xi,xj):
    x1,y1 = xi
    x2,y2 = xj
    dist = np.arccosh(np.cosh(y1)*np.cosh(x2-x1)*np.cosh(y2)-np.sinh(y1)*np.sinh(y2))
    return dist

def scale_matrix(d,new_max):
    d_new = np.zeros(d.shape)

    t_min,r_min,r_max,t_max = 0,0,np.max(d),new_max

    for i in range(d.shape[0]):
        for j in range(i):
            m = ((d[i][j]-r_min)/(r_max-r_min))*(t_max-t_min)+t_min
            d_new[i][j] = m
            d_new[j][i] = m
    return d_new




def output_hyperbolic(X,G):
    import networkx as nx
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

    Y = myMDS(d)
    Y.solve(15)
    print(Y.calc_distortion())
    output_euclidean(Y.X)


    best_X = []
    best_score = 10000000

    for i in range(10):
        Y = myHMDS(d,init_pos=Y.X)
        Y.solve(50)
        if Y.calc_stress() < best_score:
            best_score = Y.calc_distortion()
            best_X = Y.X
            print('got better')
        print(i)
    output_hyperbolic(best_X,G,0)
    print(best_score)
