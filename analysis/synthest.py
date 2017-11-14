import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import ipdb
import process_datasets as proc
from sensitivity_analysis import SensitivityAnalysis
import putils as pu
import networkx as nx


#NG, GT, labels  = proc.get_data('ecoli',0.475)
#GT = proc.cluster_matrix(2047,1,0.1,0.1,'constant',0.5)

# 2047 nodes since 2^10 is 1024 * 2 branches
erG = nx.erdos_renyi_graph(2047, 0.1)
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

nx.draw(G, node_size=30)
plt.show()

