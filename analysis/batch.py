"""
UTF-8
"""

import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import ipdb
import process_datasets as proc
from sensitivity_analysis import SensitivityAnalysis
import putils as pu


def search_dset(dataset, sigma):
    """
    Search for a .npz file into data/ folder then if exists it returns the dictionary with NG, GT, bounds
    :param dataset: of the dataset
    :param sigma: sigma of the gaussian kernel
    :returns: None if no file or the dictionary
    """
    path = "data/npz/"
    for f in os.listdir(path):
        if f == f"{dataset}_{sigma:.5f}.npz":
            return np.load(path+f)
    return None



# Argparse
#parser = argparse.ArgumentParser(description="Analysis of Szemeredi algorithm implementation")
#parser.add_argument("dataset", help="Dataset name", type=str)
#parser.add_argument("sigma", help="Sigma of Gaussian kernel", type=float)
#args = parser.parse_args()
#dataset = args.dataset
#sigma = args.sigma

dataset = 'ecoli'
sigma = 0.475

print("[+] Searching for .npz file")
data = search_dset(dataset, sigma)
first_time = False
if not data:
    first_time = True
    print("    No data: Generating ...")

    if dataset == 'XPCA':
        NG, GT, labels = proc.get_XPCA_data(sigma, 0)
    elif dataset == 'GColi1':
        NG, GT, labels = proc.get_GColi1_data()
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
s.find_bounding_epsilons()


if first_time:
    np.savez(f"data/npz/{dataset}_{sigma:.5f}.npz", NG=NG, GT=GT, bounds=s.bounds, labels=labels)
    print("    Data saved")


print("[+] Finding partitions")
kec = s.find_partitions()


print("[+] Performing threshold analysis")
for k in kec.keys():
    print(f"  k: {k}")
    classes = kec[k][1]
    dists = s.thresholds_analysis(classes, k, s.knn_voting_system)
    plt.plot(s.thresholds, dists, label=f"k={k}")

ng_dist = s.knn_voting_system(s.NG)
print(f"  NG:\n    {ng_dist:.5f}")
ng_dists = [ng_dist]*len(s.thresholds)
plt.plot(s.thresholds, ng_dists, label="NG")

# Plot
plt.title(f"{dataset}_{sigma:.5f}")
plt.ylabel('Distance')
plt.xlabel('Reconstruction Threshold')
plt.legend(loc='lower right')

plt.show()





