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
    with open('data/revision_exp1_2.pkl', 'rb') as myfile:
        final = pickle.load(myfile)
    myfile.close()


    classic_hist = np.zeros(len(final[0]['stress_hist_classic'][0]))
    for i in range(len(final[0]['stress_hist_classic'])):
        for j in range(len(classic_hist)):
            classic_hist[j] += final[0]['stress_hist_classic'][i][j]
            print(final[0]['stress_hist_classic'][i][j])
    classic_hist = np.array(classic_hist)/j

    stochastic_hist = np.zeros(len(final[0]['stress_hist_stochastic'][0]))
    for i in range(len(final[0]['stress_hist_stochastic'])):
        for j in range(len(stochastic_hist)):
            stochastic_hist[j] += final[0]['stress_hist_stochastic'][i][j]
    stochastic_hist = np.array(stochastic_hist)/j

    #classic_hist = final[0]['stress_hist_classic'][0]
    #stochastic_hist = final[0]['stress_hist_stochastic'][0]

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
    #plt.ylim(100,600)
    plt.suptitle("Stress curves on 1138bus")
    plt.legend()

    #G = [nx.grid_graph([5,5]),nx.random_tree(50),get_hyperbolic_graph(50),nx.ladder_graph(25),nx.drawing.nx_agraph.read_dot('input.dot')]

    plt.show()
    #plt.savefig("figs/updated_sphereveuclid.png")


def plot_exp_2():
        with open('data/revision_exp_smart_init.pkl', 'rb') as myfile:
            final = pickle.load(myfile)
        myfile.close()

        n = len(final[0]['stress_hist_random'])

        classic_hist = np.zeros(len(final[2]['stress_hist_random'][0]))
        for i in range(n):
            for j in range(len(classic_hist)):
                classic_hist[j] += final[1]['stress_hist_random'][i][j]
        classic_hist = np.array(classic_hist)/j

        stochastic_hist = np.zeros(len(final[2]['stress_hist_smart'][0]))
        for i in range(n):
            for j in range(len(stochastic_hist)):
                stochastic_hist[j] += final[1]['stress_hist_smart'][i][j]
        stochastic_hist = np.array(stochastic_hist)/j



        x1 = np.arange(len(classic_hist))
        x2 = np.arange(len(stochastic_hist))
        from modHMDS import calc_stress
        print(final[1]['random_layout'])
        print(calc_stress(final[1]['random_layout'],final[1]['info'][1],np.ones(final[1]['info'][1].shape)))

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


def plot_time():
    import csv
    with open('data/time_exp1_stoch.csv', newline='') as csvfile:
        spamwriter = csv.reader(csvfile, delimiter=',', quotechar='|')
        sgd = [np.array(row) for row in spamwriter]
    csvfile.close()

    with open('data/time_exp1_classic.csv', newline='') as csvfile:
        spamwriter = csv.reader(csvfile, delimiter=',', quotechar='|')
        classic = [np.array(row) for row in spamwriter]
    csvfile.close()

    sgd = np.array(sgd).astype(float)
    classic = np.array(classic).astype(float)

    sgd = [np.mean(x) for x in sgd]
    classic = [np.mean(x) for x in classic]

    x1 = [i for i in range(20,501,20)]
    x2 = x1[:18] + x1[19:]
    print(classic)

    plt.plot(x1,sgd,label='SGD time (seconds)')
    plt.plot(x2,classic,label='GD time (seconds)')
    plt.xlabel('|V| (|E| = 3|V|)')
    plt.ylabel('Time (seconds)')
    plt.suptitle("Hyperbolic GD vs. SGD Average runtime")
    plt.legend()
    plt.savefig("Runtime.eps")


plot_time()
