import numpy as np
import matplotlib.pyplot as plt 
from sklearn import metrics
import pdb

import sys
sys.path.insert(1, '/home/lakj/Documenti/university/thesis/code/dense_graph_reducer_forked/analysis')
import real_data as rd



NG, GT, tot_dim = rd.get_UCI_data('ecoli', 0.475)


n = GT.shape[0]

ind_vector = np.zeros(n)
ind_vector[0:144] = 1
ind_vector[144:221] = 2
ind_vector[221:223] = 3
ind_vector[223:225] = 4
ind_vector[225:260] = 5
ind_vector[260:280] = 6
ind_vector[280:285] = 7
ind_vector[285:336] = 8

labels = np.zeros(n)
k = 5
h = [GT[219,:], GT[220,:], GT[221,:], GT[222,:], GT[223,:], GT[224,:], GT[225,:], GT[226,:]]

plt.imshow(GT[219:226,:])
plt.show(block=False)

i = 219 
for row in [GT[219,:], GT[220,:], GT[221,:], GT[222,:], GT[223,:], GT[224,:], GT[225,:], GT[226,:]]:

    #k_indices = row.argsort()[-k:][::-1]
    pdb.set_trace()
    
    a = row.argsort()[-k:]
    b = row[a] > 0
    k_indices = a[b]

    if len(k_indices) == 0:
        k_indices = row.argsort()[-1:]

    candidate_lbl = np.bincount(ind_vector[k_indices].astype(int)).argmax()
    labels[i] = candidate_lbl
    i += 1

    ars = metrics.adjusted_rand_score(ind_vector, labels)

    print(labels)
