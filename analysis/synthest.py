import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import ipdb
import process_datasets as proc
from sensitivity_analysis import SensitivityAnalysis
import putils as pu
import networkx as nx


# TODO togliere adj_mat reconstructed 

#NG, GT, labels  = proc.get_data('ecoli',0.475)
#GT = proc.cluster_matrix(2047,1,0.1,0.1,'constant',0.5)


g = proc.synthetic_regular_partition(32, 0.65)
plt.imshow(g)
plt.show()

k = 32
thresh = 0

classes = np.repeat([x for x in range(1,33)], 100)

# New matrix
reconstructed_mat = np.zeros((len(classes), len(classes)), dtype='float32')

for r in range(2, k + 1):
    r_nodes = classes == r
    for s in range(1, r):
        s_nodes = classes == s
        bip_sim_mat = g[np.ix_(index_map[0], index_map[1])]
        n = bip_sim_mat.shape[0]
        bip_density = bip_sim_mat.sum() / (n ** 2.0)
        # Put edges if above threshold
        if bip_density > thresh:
            reconstructed_mat[np.ix_(r_nodes, s_nodes)] = reconstructed_mat[np.ix_(s_nodes, r_nodes)] = bip_density
np.fill_diagonal(reconstructed_mat, 0.0)

print(reconstructed_mat)

"""
# 2047 nodes since 2^10 is 1024 * 2 branches
erG = nx.erdos_renyi_graph(2047, 0.5)
erNG = nx.to_numpy_array(erG)
aux = erNG > 0
erNG[aux] = 0.5

G = nx.balanced_tree(2,10)
NG = nx.to_numpy_array(G)

NG += erNG
idxs = NG > 1
NG[idxs] = 1

plt.imshow(NG)
plt.show()


data = {}
labels = []
data['NG'] = NG
data['GT'] = NG
data['labels'] = labels
data['bounds'] = []


s = SensitivityAnalysis(data)
bounds = s.find_bounds()
partitions = s.find_partitions()

print(partitions.keys())

nx.draw(G, node_size=10)
plt.show()

"""
