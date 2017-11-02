"""
UTF-8
author: francesco-p
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering 
from sklearn import metrics
import pdb
import math

import sys
sys.path.insert(0, '/home/lakj/Documenti/university/thesis/code/dense_graph_reducer_forked/graph_reducer/')
sys.path.insert(1, '/home/lakj/Documenti/university/thesis/code/dense_graph_reducer_forked/analysis/')
import szemeredi_lemma_builder as slb
import process_datasets as pd


def get_NG_t(name):
    # TODO bisogna trovare sigma prima
    data = sp.loadmat(f"/home/lakj/Documenti/university/thesis/code/dense_graph_reducer_forked/analysis/pymat/data/NG_ts/{name}_NG_t.mat")
    return data['NG_t']


def create_UCI_graphs(name, sigma):
    NG, GT, tot_dim = pd.get_UCI_data(name, sigma)
    title = f'UCI_{name}_dataset_sigma_{sigma:.10f}'
    return NG, GT, title, tot_dim


def knn_voting_system(G, ind_vector, k):
    n = G.shape[0]
    labels = np.zeros(n)
    i = 0

    for row in G:
        #k_indices = row.argsort()[-k:][::-1]
        aux = row.argsort()[-k:]
        aux2 = row[aux] > 0
        k_indices = aux[aux2]
        
        if len(k_indices) == 0:
            k_indices = row.argsort()[-1:]

        candidate_lbl = np.bincount(ind_vector[k_indices].astype(int)).argmax()
        labels[i] = candidate_lbl
        i += 1

    ars = metrics.adjusted_rand_score(ind_vector, labels)

    return ars 

#pdb.set_trace()
dset = 'userknowledge'
#sigma = 0.475
sigma = float(sys.argv[1])

# Szemeredi algorithm constants
kind = "alon"
is_weighted = True
random_initialization = True
random_refinement = False
drop_edges_between_irregular_pairs = True
compression = 0.05

if dset == 'ecoli':
    #sigma = 0.475
    # ECOLI IND VECTOR
    ind_vector = np.zeros(336)
    ind_vector[0:144] = 1
    ind_vector[144:221] = 2
    ind_vector[221:223] = 3
    ind_vector[223:225] = 4
    ind_vector[225:260] = 5
    ind_vector[260:280] = 6
    ind_vector[280:285] = 7
    ind_vector[285:336] = 8
elif dset == 'ionosphere':
    #sigma = 0.12
    # iono IND VECTOR
    ind_vector = np.zeros(351)
    ind_vector[0:126] = 1
    ind_vector[126:351] = 2
elif dset == 'userknowledge':
    #sigma = 0.42
    ind_vector = np.zeros(258)
    ind_vector[0:64] = 1
    ind_vector[64:152] = 2
    ind_vector[152:235] = 3
    ind_vector[235:258] = 4
elif dset == 'spect-singleproton':
    #sigma = 1.5  
    ind_vector = np.zeros(267)
    ind_vector[0:56] = 1
    ind_vector[56:267] = 4

NG, GT, title, tot_dim = create_UCI_graphs(dset, sigma) 


def find_trivial_epsilon(epsilon1, epsilon2, k_min, tolerance):
    step = (epsilon2 - epsilon1)/2.0
    if step < tolerance:
        return epsilon2
    else:
        epsilon_middle = epsilon1 + step
        print(f"|{epsilon1:.6f}-----{epsilon_middle:.6f}------{epsilon2:.6f}|", end=" ")
        srla = slb.generate_szemeredi_reg_lemma_implementation(kind, NG, epsilon_middle, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)
        regular, k, reduced_matrix = srla.run(iteration_by_iteration=False, verbose=False, compression_rate=compression)
        print(f"{k} {regular}")
        if regular:
            if k==k_min:
                del srla
                return find_trivial_epsilon(epsilon1, epsilon_middle, k_min, tolerance)
            if k>k_min:
                del srla
                return find_trivial_epsilon(epsilon_middle, epsilon2, k_min, tolerance)
            else:
                del srla
                return -1
        else:
            del srla
            return find_trivial_epsilon(epsilon_middle, epsilon2, k_min, tolerance)

def find_edge_epsilon(epsilon1, epsilon2, tolerance):
    step = (epsilon2 - epsilon1)/2.0
    if step < tolerance:
        return epsilon2
    else:
        epsilon_middle = epsilon1 + step
        print(f"|{epsilon1:.6f}-----{epsilon_middle:.6f}------{epsilon2:.6f}|", end=" ")
        srla = slb.generate_szemeredi_reg_lemma_implementation(kind, NG, epsilon_middle, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)
        regular, k, reduced_matrix = srla.run(iteration_by_iteration=False, verbose=False, compression_rate=compression)
        print(f"{k} {regular}")
        if regular:
            del srla
            return find_edge_epsilon(epsilon1, epsilon_middle, tolerance)
        else:
            del srla
            return find_edge_epsilon(epsilon_middle, epsilon2, tolerance)

print("Finding trivial epsilon...")
epsilon2 = find_trivial_epsilon(0.1,0.9, 2, 0.00001)
print("Trivial epsilon candidate: {0:.6f}".format(epsilon2))
print("Finding edge epsilon...")
epsilon1 = find_edge_epsilon(0.1,epsilon2,0.00001)
print("Edge epsilon candidate: {0:.6f}".format(epsilon1))

epsilons = [epsilon1]

# Try 20 different epsilons inside the bounds
tries = 20.0
offs = (epsilon2 - epsilon1) / tries
for i in range(1, int(tries)+1):
    epsilons.append(epsilon1 + (i*offs))

# Dictrionary to hold all the different distances of each different partition k
thresholds = np.arange(0, 1.05, 0.05)

knn_values = [3, 5, 10, 15, int(math.sqrt(GT.shape[0]))]

knn_three = {}
for knn in knn_values:

    k_dist = {}

    # Points structure
    points = [] 

    # NG idx 
    distance = knn_voting_system(NG, ind_vector, knn)
    print(f"NG :     ars_idx: {distance:.4}")
    points.append(distance) 

    # NG_t idx 
    NG_t = get_NG_t(dset)
    distance = knn_voting_system(abs(NG_t-1), ind_vector, knn)
    print(f"NG_t :     ars_idx: {distance:.4}")
    points.append(distance) 

    all_dists = {} 
    for epsilon in epsilons:

        srla = slb.generate_szemeredi_reg_lemma_implementation(kind, NG, epsilon, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)
        regular, k, reduced_matrix = srla.run(iteration_by_iteration=False, verbose=False, compression_rate=compression)

        if (k not in k_dist) and regular and k!=2:
            print(f"FOUND PARTITION K {k} knn{knn}")
            dsts = [] 
            print("SZE:")
            for thresh in thresholds:
                
                SZE_rec = srla.reconstruct_original_mat(thresh)
                distance = knn_voting_system(SZE_rec, ind_vector, knn) 

                print(f"    ars_idx:{distance:.4} knn:{knn}  k: {k}  thresh: {thresh:.2}")
                all_dists[distance] = (k, thresh)
                
                dsts.append(distance)

        k_dist[k] = dsts 

    # LOST
    kmax = all_dists[max(all_dists.keys())][0]
    tmax = all_dists[max(all_dists.keys())][1]

    sze_max = max(all_dists.keys())
    points.append(sze_max)

    knn_three[knn] = points

ngs = []
ngts = []
szes = []
for k_n in knn_three.keys():
    a = knn_three[k_n]
    ngs.append(a[0])
    ngts.append(a[1])
    szes.append(a[2])

#print(ngs)
#print(ngts)
#print(szes)


# Generate the plot TODO
plt.plot(knn_values, ngs, label="NG")
plt.plot(knn_values, ngts, label="abs(NG_t-1)")
plt.plot(knn_values, szes, label="best SZE")

ttl = f"{title}"
plt.title(f"{ttl}")
plt.ylabel('adj_rnd_idx')
plt.xlabel('k of knn')
plt.xticks(knn_values, knn_values)
plt.legend(loc='upper right')
plt.savefig(f"./imgs/{ttl}_points.png")

#plt.show()

# Amplitude Commute Time Kernel
#NG_t = get_NG_t(dset)
#ars = metrics.adjusted_rand_score(kmeans.labels_, pred)
#mis = metrics.adjusted_mutual_info_score(kmeans.labels_, pred)
#print(f"ACT: {ars:.4} {mis:.4}")


# Save the bounds of the analysis
#with open("./imgs/bounds.txt", "a") as bounds_file:
#    bounds_file.write(f"{title}_{epsilon1}_{epsilon2}\n")


# Adds NOISY GRAPH line
#ng_dist = L2_distance(GT, NG)/tot_dim
#k_dist['NG'] = [ng_dist]*len(thresholds) 


# Save a dictionary of the results (we can plot it directly with plot.py)
#name = f"./npy/{title}.npy"
#np.save(name , k_dist)
"""
def indicator_vector(GT):
    # non va TODO 
    n = GT.shape[0]
    labels = np.zeros(n)

    col = 0
    row = 0
    start = 0
    b = 1
    found = False
    while col != n:
        while row != n:
            if GT[row, col] == 0:
                found = True
                labels[start:start+row] = b
                start = row
                b += 1
            else:
                found = False
            
            if found:
                row = 0
                break
            else:
                row += 1
        if found:
            col = start
        else:
            col += 1

    return labels

"""
