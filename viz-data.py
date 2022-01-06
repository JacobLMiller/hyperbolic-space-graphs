import pickle
import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt


def stress(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += pow(np.linalg.norm(X[i]-X[j])-d[i][j],2)
    return stress

def choose(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

def dist(X,d):
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += abs(np.linalg.norm(X[i]-X[j])-d[i][j])/d[i][j]
    return stress/choose(len(X),2)

def plot_exp_1():
    with open('data/revision_exp1.pkl', 'rb') as myfile:
        final = pickle.load(myfile)
    myfile.close()

    with open('data/revision_exp2.pkl', 'rb') as myfile:
        stochdata = pickle.load(myfile)
    myfile.close()

    classic_hist = np.zeros(len(final[2]['stress_hist_classic'][0]))
    for i in range(len(final[2]['stress_hist_classic'])):
        for j in range(len(classic_hist)):
            classic_hist[j] += final[2]['stress_hist_classic'][i][j]
    classic_hist /= 5

    stochastic_hist = np.zeros(len(stochdata[2]['stress_hist_stochastic'][0]))
    for i in range(len(stochdata[2]['stress_hist_stochastic'])):
        for j in range(len(stochastic_hist)):
            stochastic_hist[j] += stochdata[2]['stress_hist_stochastic'][i][j]
    stochastic_hist /= len(stochdata[2]['stress_hist_stochastic'])

    #classic_hist = final[0]['stress_hist_classic'][0]
    #stochastic_hist = stochdata[0]['stress_hist_stochastic'][0]

    x1 = 1+np.arange(len(classic_hist))
    x2 = 1+np.arange(len(stochastic_hist))
    print(stochastic_hist)

    plt.plot(x1, classic_hist, label="Classic Average")
    plt.plot(x2, stochastic_hist,label="Stocastic Average")

    #plt.plot(x, hyper_distortion, label = "Hyperbolic Distortion")
    plt.xlabel("Iteration")
    plt.ylabel("Stress")
    plt.xlim()
    #plt.yscale('log')
    #plt.ylim(0,1)
    plt.suptitle("Stress curves on 1138bus")
    plt.legend()

    #G = [nx.grid_graph([5,5]),nx.random_tree(50),get_hyperbolic_graph(50),nx.ladder_graph(25),nx.drawing.nx_agraph.read_dot('input.dot')]

    plt.show()
    #plt.savefig("figs/updated_sphereveuclid.png")


def plot_exp_2():
        with open('data/revision_exp3.pkl', 'rb') as myfile:
            final = pickle.load(myfile)
        myfile.close()

        with open('data/revision_exp2.pkl', 'rb') as myfile:
            stochdata = pickle.load(myfile)
        myfile.close()

        classic_hist = np.zeros(len(final[2]['stress_hist_stochastic'][0]))
        for i in range(len(final[2]['stress_hist_stochastic'])):
            for j in range(len(classic_hist)):
                classic_hist[j] += final[3]['stress_hist_stochastic'][i][j]
        classic_hist /= 5

        stochastic_hist = np.zeros(len(stochdata[2]['stress_hist_stochastic'][0]))
        for i in range(len(stochdata[2]['stress_hist_stochastic'])):
            for j in range(len(stochastic_hist)):
                stochastic_hist[j] += stochdata[3]['stress_hist_stochastic'][i][j]
        stochastic_hist /= len(stochdata[2]['stress_hist_stochastic'])
        stochastic_hist = stochastic_hist[:15]

        #classic_hist = final[0]['stress_hist_classic'][0]
        #stochastic_hist = stochdata[0]['stress_hist_stochastic'][0]

        x1 = np.arange(len(classic_hist))
        x2 = np.arange(len(stochastic_hist))
        print(stochastic_hist)

        plt.plot(x1, classic_hist, label="Smart init")
        plt.plot(x2, stochastic_hist,label="Random Init")

        #plt.plot(x, hyper_distortion, label = "Hyperbolic Distortion")
        plt.xlabel("Iteration")
        plt.ylabel("Stress")
        plt.xlim()
        #plt.yscale('log')
        #plt.ylim(0,1)
        plt.suptitle("Stress curves on 1138bus")
        plt.legend()

        #G = [nx.grid_graph([5,5]),nx.random_tree(50),get_hyperbolic_graph(50),nx.ladder_graph(25),nx.drawing.nx_agraph.read_dot('input.dot')]

        plt.show()

        smart = [final[2]['stress_hist_stochastic'][i][-1] for i in range(len(final[0]['stress_hist_stochastic']))]
        print(smart)
        random = [stochdata[2]['stress_hist_stochastic'][i][-1] for i in range(len(final[0]['stress_hist_stochastic']))]

        from scipy.stats import ttest_ind
        stat, p = ttest_ind(smart,random)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
        	print('Same distributions (fail to reject H0)')
        else:
        	print('Different distributions (reject H0)')

plot_exp_2()
