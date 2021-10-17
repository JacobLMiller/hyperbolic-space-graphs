import scipy.io
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

Data = []

with open('data/optimize-eta.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        Data.append([float(i) for i in row[0].split(',')])


Data = np.asarray(Data)
#x = np.asarray([i for i in range(10,100)])
#classicTime = Data[:,0]
#stochasticTime = Data[:,1]
x = Data[:,0]
y = Data[:,1]

#euclid_time = np.array([Data[i][0] for i in range(Data.shape[0])])
#euclid_distortion = np.array([Data[i][1] for i in range(Data.shape[0])])
#hyper_time = np.array([Data[i][2] for i in range(Data.shape[0])])
#hyper_distortion = np.array([Data[i][3] for i in range(Data.shape[0])])

#plt.plot(x, euclid_time, label = "Euclidean Time")
#plt.plot(x, hyper_time, label = "Hyperbolic Time")
plt.plot(x, y, label = "Distortion")
#plt.plot(x, stochasticTime, label = "Distortion")

#plt.plot(x, hyper_distortion, label = "Hyperbolic Distortion")
plt.xlabel("Scale factor")
plt.ylabel("Distortion")
plt.xlim()
plt.suptitle("Euclidean scale")
plt.legend()

plt.show()
#plt.savefig("figures/scale_invariance_euclid_triangle_grid.png")
