"""
UTF-8
dsets = [("column3C", 0.0004), ("ecoli", 0.475), ("ionosphere", 0.12), ("iris", 0.02), ("userknowledge", 0.42), ("spect-singleproton", 1.5)]
worst = ["indian-liver", "pop-failures", "spect-singleproton", "spect-test"]
"""

import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import ipdb
import process_datasets as proc
from sensitivity_analysis import SensitivityAnalysis


def search_dset(filename):
    """
    Search for a .npz file into data/ folder then if exists it returns the dictionary with NG, GT, bounds
    :param dataset: of the dataset
    :param sigma: sigma of the gaussian kernel
    :returns: None if no file or the dictionary
    """
    path = "data/npz/"
    for f in os.listdir(path):
        if f == filename+".npz":
            return np.load(path+f)
    return None


# Argparse
parser = argparse.ArgumentParser(description="Analysis of Szemeredi algorithm implementation")
parser.add_argument("dataset", help="Dataset name", type=str)
parser.add_argument("sigma", help="Sigma of Gaussian kernel", type=float)
parser.add_argument("--save", help="Save image instead of showing it", action="store_true")
args = parser.parse_args()
dataset = args.dataset
sigma = args.sigma
filename = f"{dataset}_{sigma:.5f}"

#dataset = 'ecoli'
#sigma = 0.475

print("[+] Searching for .npz file")
data = search_dset(filename)
first_time = False
if not data:
    first_time = True
    print("    No data: Generating ...")

    if dataset == 'XPCA':
        NG, GT, labels = proc.get_XPCA_data(sigma, 0)
    elif dataset == 'GCoil1':
        NG, GT, labels = proc.get_GCoil1_data()
    elif dataset == 'custom':
        NG, GT, labels  = proc.custom_cluster_matrix(5000, [3000, 1000, 500, 500], 0, 0)
    else:
        NG, GT, labels = proc.get_data(dataset, sigma)

    data = {}
    data['NG'] = NG
    data['GT'] = GT
    data['bounds'] = []
    data['labels'] = labels
    print("    Done!")
else:
    print("[+] Data found")


# Start analysis
s = SensitivityAnalysis(data)

print("[+] Finding bounding epsilon")
bounds = s.find_bounds()


if first_time:
    np.savez(f"data/npz/{filename}.npz", NG=NG, GT=GT, bounds=bounds, labels=labels)
    print("    Data saved")

print("[+] Finding partitions")
kec = s.find_partitions()


print("[+] Performing threshold analysis")
thresholds = np.arange(0, 1.0, 0.05)
min_d = 0
min_k = -1

for k in kec.keys():
    print(f"  k: {k}")
    classes = kec[k][1]
    dists = s.thresholds_analysis(classes, k, thresholds, s.L2_metric)
    plt.plot(thresholds, dists, label=f"k={k}")

ng_dist = s.L2_metric(s.NG)
print(f"  NG:\n    {ng_dist:.5f}")
ng_dists = [ng_dist]*len(thresholds)
plt.plot(thresholds, ng_dists, label="NG")

# Plot
plt.title(filename)
plt.ylabel('L2 Distance')
plt.xlabel('Reconstruction Threshold')
plt.legend(loc='lower right')

if args.save:
    plt.savefig(f"{filename}.png")

plt.show()

