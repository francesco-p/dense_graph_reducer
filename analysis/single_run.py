from sensitivity_analysis import SensitivityAnalysis
import numpy as np
import multiprocessing
import putils as pu

#dsets = [("column3C", 0.0004), ("ecoli", 0.475), ("ionosphere", 0.12), ("iris", 0.02), ("userknowledge", 0.42), ("spect-singleproton", 1.5)]

#dataset = 'XPCA'
#sigma = 0.0124
#epsilon = 0.7

#filename = f"{dataset}_{sigma:.5f}"
#data = search_dset(filename)

"""
def work(pid, densities):
    pu.idxprint(f"[PID: {pid} ] - densities: {densities}", pid)
    for d in densities:
        filename = f"./data/synthetic_graphs/1000/1000_{d:.2f}_1.npy"
        pu.idxprint(f"[PID: {pid} ] - {filename} loaded", pid)
        G = np.load(filename)

        data = {}
        data['NG'] = G
        data['GT'] = G
        data['bounds'] = []
        data['labels'] = []
        sa = SensitivityAnalysis(data)
        sa.verbose = False
        bounds = sa.find_bounds()
        pu.idxprint(f"[PID: {pid} ] Bounds found: {bounds}", pid)
        kec = sa.find_partitions()
        pu.idxprint(f"[PID: {pid} ] Partitions found: {kec}", pid)

#######################################################################
#######################################################################

procs = 4    #Number of processes to create
jobs = []
densities = np.arange(0.5, 1, 0.05)[::-1]
for pid in range(0, procs):
    denses = np.arange(0.5+(0.05*pid), 1, 0.05*procs)[::-1]
    process = multiprocessing.Process(target=work, args=(pid, denses))
    jobs.append(process)

for j in jobs:
        j.start()
for j in jobs:
        j.join()

#denses = np.arange(0.5, 1, 0.05)[::-1]
#work(0, denses)

print("Work completed")
"""

n = 10000
d = 0.8

G = np.tril(np.random.random((n, n)) < d, -1).astype('int8')
G += G.T

data = {}
data['NG'] = G
data['GT'] = G
data['bounds'] = []
data['labels'] = []
sa = SensitivityAnalysis(data)
bounds = sa.find_bounds()
print(f"bounds: bounds")
kec = sa.find_partitions()
print(f"partitions: {kec.keys()}")

