"""
File: batch2.py
Description:
 1. fix density d and size of the graph n
 2. process the 500 datasets
 3. find bounds and partitions
 4. for each partition then take the one with the maximum idx
 5. reconstruct the matrix with threshold 0
 6. compute the measures
 7. plot foreach density
Coding: UTF-8
Author: lakj
"""
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import ipdb
import process_datasets as proc
from sensitivity_analysis import SensitivityAnalysis
import multiprocessing
import io

def work(pid, n, d, dset_ids):

    for dset_id in dset_ids:
        ### 2. ###
        filename = f"./data/synthetic_graphs/{n}/{n}_{d:.2f}_{dset_id}.npz"

        data = proc.search_dset(filename, synth=True)

        ### 3. ###
        s = SensitivityAnalysis(data)

        bounds = s.find_bounds()
        if not s.bounds:
            np.savez_compressed(filename, G=data['G'], bounds=bounds)

        kec = s.find_partitions()

        if kec == {}:
            print(f"[x] {n}_{d:.2f}_{dset_id}.npy")

        else:

            ### 4. ###
            max_idx = -1
            max_k = -1

            for k in kec.keys():
                if kec[k][2] > max_idx:
                    max_k = k
                    max_idx = kec[k][2]

            #print(f"[+] Partition with the highest sze_idx k: {max_k} idx: {kec[max_k][2]:.4f}")

            ### 5. ###
            sze_rec = s.reconstruct_mat(0, kec[max_k][1], max_k)

            ### 6. ###
            dist = s.L2_metric(sze_rec)
            print(f"[OK] {n}_{d:.2f}_{dset_id}.npy l2d:{dist:.4f} sze_idx:{kec[max_k][2]:.4f}")
            with io.open(f'./data/synthetic_graphs/csv/{n}.csv', 'a') as f:
                f.write(f"{n},{d:.2f},{dset_id},{dist:.4f},{kec[max_k][2]:.4f},{s.bounds[0]:.5f},{s.bounds[1]:.5f},1\n")

####################
#denses = np.arange(0.5+(0.05*pid), 1, 0.05*procs)[::-1]

procs = 4    #Number of processes to create
n = 400
n_graphs = 500
densities = np.arange(0.5, 1, 0.05)
with io.open(f'./data/synthetic_graphs/{n}.csv', 'w') as f:
    f.write(f"n,density,dset_id,measure,sze_idx,edge,trivial,refinement\n")
for d in densities:
    jobs = []
    for pid in range(0, procs):
        dset_ids = range(1+pid, n_graphs+1, procs)
        process = multiprocessing.Process(target=work, args=(pid, n, d, dset_ids))
        jobs.append(process)

    for j in jobs:
            j.start()
    for j in jobs:
            j.join()

"""
raise IOError("fuck")
### 7. ###
data_to_plot = []
for d in densities:
    with io.open(f'{n}_{d:.2f}.log', 'r') as f:
        values = []
        for line in f:
            values.append(float(lines))
        data_to_plot.append(values)

fig = plt.figure(1, figsize=(9, 6))
plt.ylabel("L2 distance")
plt.xlabel('Density')
plt.title(f"N = {n}")
ax = fig.add_subplot(111)
ax.boxplot(data_to_plot)
ax.set_xticklabels(densities)
plt.show()

"""



