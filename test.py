import scipy.io
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

Data = []

with open('data/randomization.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        Data.append([float(i) for i in row[0].split(',')])

otherData = []
with open('data/trees.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        otherData.append([float(i) for i in row[0].split(',')])


otherData = np.array(otherData)
Data = np.array(Data)
print(Data)
print(otherData)

#x = np.asarray([i for i in range(10,100)])
#classicTime = Data[:,0]
#stochasticTime = Data[:,1]
#x = np.array([i for i in range(len(Test))])
#y = Test
#z = otherData[10,:len(Test)]/50
#z0 = Data[14]/50
#euclid_time = np.array([Data[i][0] for i in range(Data.shape[0])])
#euclid_distortion = np.array([Data[i][1] for i in range(Data.shape[0])])
#hyper_time = np.array([Data[i][2] for i in range(Data.shape[0])])
#hyper_distortion = np.array([Data[i][3] for i in range(Data.shape[0])])
#[nx.grid_graph([10,10]),nx.random_tree(50),get_hyperbolic_graph(50),nx.les_miserables_graph(),nx.drawing.nx_agraph.read_dot('SGD/input.dot')]

x = np.asarray([i for i in range(1000)])
y = Data[1]
z = Data[6]
z0 = Data[11]
#z0 = Data[11,:29]/50

#plt.plot(x, euclid_time, label = "Euclidean Time")
#plt.plot(x, hyper_time, label = "Hyperbolic Time")
plt.plot(x, y, label="Random reshuffling")
plt.plot(x, z,label="Shuffle indices")
plt.plot(x,z0, label = "Sample with replacement")

#plt.plot(x, hyper_distortion, label = "Hyperbolic Distortion")
plt.xlabel("# of nodes")
plt.ylabel("Average Stress")
plt.xlim()
plt.ylim()
plt.suptitle("50 node random tree")
plt.legend()

#G = [nx.grid_graph([5,5]),nx.random_tree(50),get_hyperbolic_graph(50),nx.ladder_graph(25),nx.drawing.nx_agraph.read_dot('input.dot')]

#plt.show()
plt.savefig("SGD/figs/tree_random.eps",format='eps')
