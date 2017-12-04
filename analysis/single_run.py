from sensitivity_analysis import SensitivityAnalysis
import matplotlib.pyplot as plt
import numpy as np
import os
import process_datasets as pd
import ipdb

#dsets = [("column3C", 0.0004), ("ecoli", 0.475), ("ionosphere", 0.12), ("iris", 0.02), ("userknowledge", 0.42), ("spect-singleproton", 1.5)]

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

#dataset = 'XPCA'
#sigma = 0.0124
#epsilon = 0.7

#filename = f"{dataset}_{sigma:.5f}"
#data = search_dset(filename)

#NG, GT, labels  = pd.custom_cluster_matrix(60, [25, 15, 11, 9], 0, 0)
NG, GT, labels  = pd.custom_cluster_matrix(5000, [3000, 1000, 500, 500], 0, 0)

data = {}
data['NG'] = NG
data['GT'] = GT
data['bounds'] = []
data['labels'] = labels
sa = SensitivityAnalysis(data)
bounds = sa.find_bounds()
print(bounds)
kec = sa.find_partitions()
print(kec)

