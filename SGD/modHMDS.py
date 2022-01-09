import numpy as np
#import tensorflow as tf
import math
import random
import itertools


from numba import jit

@jit(nopython=True)
def geodesic(u,v):
    x1,y1 = u
    x2,y2 = v
    return np.arccosh(np.cosh(y1)*np.cosh(x2-x1)*np.cosh(y2)-np.sinh(y1)*np.sinh(y2))

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
def stoch_solver(X,d,w,indices,schedule,num_iter=15,epsilon=1e-3,debug=False):
    step = 0.1
    shuffle = random.shuffle
    if debug:
        stress_hist = []

    for count in range(num_iter):
        for i,j in indices:
            X[i],X[j] = satisfy(X[i],X[j],d[i][j],w[i][j],step)

        step = schedule[count] if count <= len(schedule) else schedule[-1]
        shuffle(indices)
        if debug:
            stress_hist.append(calc_stress(X,d,w))
    return X

@jit(nopython=True)
def set_step(w_max,eta_max,eta_min):
    a = 1/w_max
    b = -np.log(eta_min/eta_max)/(15-1)
    step = lambda count: a/(pow(1+b*count,0.5))
    return np.array([step(count) for count in range(15)])

class HMDS:
    def __init__(self,dissimilarities,epsilon=0.1,init_pos=np.array([])):
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
        self.eta_min = epsilon/self.w_max

        self.indices = np.array(list(itertools.combinations(range(self.n), 2)))

        self.steps = set_step(self.w_max,self.eta_max,self.eta_min)


    def solve(self,debug=False):
        X = stoch_solver(self.X,self.d,self.w,self.indices,self.steps)
        self.X = X

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

def mobius(z,a,b,c,d):
    return a*z+b/c*z+d

def geodesic2(xi,xj):
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
#g = ig.Graph.Tree(500,2)
#g.write_dot('input.dot')

#G = nx.drawing.nx_agraph.read_dot('SGD/input.dot')
#G = nx.full_rary_tree(2,200)
#G = nx.hypercube_graph(3)
#G = get_hyperbolic_graph(100,alpha=1.2)
#G = nx.circular_ladder_graph(20)
#G = nx.triangular_lattice_graph(5,5)
#nx.drawing.nx_agraph.write_dot(G, "output_hyperbolic.dot")
#G = nx.random_tree(100)
#print(G.nodes())
#d = np.array(all_pairs_shortest_path(G))/1
#main()
