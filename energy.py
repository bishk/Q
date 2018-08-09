import numpy as np
from graph import load_graph, SparseGraph
from scipy.optimize import basinhopping
from GA import GA
from skopt.space import Real, Integer
from copy import deepcopy
import random
import multiprocessing as mp
from dwave_sapi2.util import qubo_to_ising
from collections import Counter
from sklearn.metrics import accuracy_score
import neal
from scipy.sparse import dok_matrix
from contextlib import closing

graph = load_graph("./graphs/graph000000.npz", SparseGraph)

def dist(vec):
    x, y, z = vec
    return np.sqrt(x**2.0 + y**2.0 + z**2.0)

def dist2d(vec):
    x, y, z = vec
    return np.sqrt(x**2.0 + y**2.0)

def geometricweight(graph, k, l, m):
    x1, y1, z1 = graph.X[k]
    x2, y2, z2 = graph.X[l]
    x3, y3, z3 = graph.X[m]
    vec1 = (x2-x1, y2-y1, z2-z1)
    vec2 = (x3-x2, y3-y2, z3-z2)
    if np.dot(vec1, vec2)/(dist(vec1)*dist(vec2)) > 0.5:
        return (np.dot(vec1, vec2)/(dist(vec1)*dist(vec2)))**100.0/(dist2d(vec1)**2.0 + dist2d(vec2)**2.0)
    return 0

def getweights(graph):
    n_hits = len(graph.X)
    n_edges = len(graph.y)
    one = np.zeros((n_edges, n_edges))
    two = np.zeros((n_edges, n_edges))
    three1 = np.zeros(n_edges)
    three2 = np.zeros((n_edges, n_edges))
    for i in range(n_edges):
        three1[i] += 1.0
        three1[i] -= 2.0*n_hits
        for j in range(i+1, n_edges):
            i1 = np.where(graph.Ri_cols == i)[0][0]
            i2 = np.where(graph.Ro_cols == i)[0][0]
            j1 = np.where(graph.Ri_cols == j)[0][0]
            j2 = np.where(graph.Ro_cols == j)[0][0]
            k = graph.Ro_rows[i2]
            l = graph.Ri_rows[i1]
            m = graph.Ri_rows[j1]
            weight = geometricweight(graph, k, l, m)
            one[i][j] = weight
            one[j][i] = one[i][j]
            if graph.Ri_rows[i1] == graph.Ri_rows[j1] or graph.Ro_rows[i2] == graph.Ro_rows[j2]:
                two[i][j] += 1.0
                two[j][i] = two[i][j]
            three2[i][j] += 2.0
    return one, two, three1, three2

def hJ(x):
    a, b, c = x[0]
    one, two, three1, three2 = getweights(graph)
    Q = dict()
    for i in range(0,len(graph.y)):
        Q[(i,i)] = -1*(-b*three1[i])
        for j in range(i+1, len(graph.y)):
            Q[(i, j)] = -0.5*(a*one[i][j] - c*two[i][j] -b*three2[i][j])
            Q[(j, i)] = Q[(i, j)]
    (h, J, x) = qubo_to_ising(Q)
    sampler = neal.SimulatedAnnealingSampler()
    hp = dict()
    for i in range(len(h)):
        hp[i] = h[i]
    response = sampler.sample_ising(hp, J, beta_range = (0.1, 10), num_reads=5)
    su = 0
    for r in response:
        su+= accuracy_score([(v+1)/2.0 for k,v in r.items()], graph.y)
    return su


paramRanges = [Real(0.00001, 100), Real(0.00001, 100), Real(0.00001, 100)]
populationSize = 1500
generations = 200
ga = GA(paramRanges, populationSize, generations)
bestParams = []
fitnessHistory = []
for g in range(generations):
    print('GENERATION', g)
    population = ga.ask()
    fitnesses = []
    args = []
    for h in range(populationSize):
        args.append([population[h]])
    with closing(mp.Pool()) as pool:
        fitnesses = pool.map(hJ, args)
        pool.terminate()
        bestFit = ga.tell(population, fitnesses, g)
        bestParams = bestFit[0]
        fitnessHistory.append(bestFit[1])
print('best params:', bestParams)
print('fitness history:', fitnessHistory)



            