import scipy.io as sp
import math
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/lakj/Documenti/university/thesis/code/dense_graph_reducer_forked/misc')
import noisyblockadjmat as nbam


def graph_from_points(x, sigma, to_remove=0):
    """
    Generates a graph (weighted graph) from a set of points x (nxd) and a sigma decay
    :param x: a numpy matrix of n times d dimension
    :param sigma: a sigma for the gaussian kernel
    :param to_remove: imbalances the last cluster
    :return: a weighted symmetric graph
    """

    n = x.shape[0]
    n -= to_remove
    w_graph = np.zeros((n,n))

    for i in range(0,n):
        copy = np.tile(np.array(x[i, :]), (i+1, 1))
        difference = copy - x[0:i+1, :]
        column = np.exp(-sigma*(difference**2).sum(1))

        #w_graph[0:i+1, i] = column 
        w_graph[0:i, i] = column[:-1] # set diagonal to 0 the resulting graph is different

    return w_graph + w_graph.T


def get_XPCA_data(sigma, to_remove):

    c_dimensions = ['XPCA']

    data = sp.loadmat("data/XPCA.mat")

    arr_feature = data['X']
    NG = graph_from_points(arr_feature, sigma, to_remove) #0.0124 00248  0.0496 e 0.0744

    # Generates the custo GT wrt the number of rows removed
    tot_dim = 10000-to_remove
    c_dimensions = [1000]*int(tot_dim/1000)
    if tot_dim % 1000:
        c_dimensions.append(tot_dim % 1000)
    GT = nbam.custom_noisy_matrix(tot_dim, c_dimensions, 0, 0)

    #plt.show(plt.imshow(NG))
    #plt.show(plt.imshow(GT))

    return NG, GT, tot_dim


def get_UCI_data(name, sigma):
    """
    Process UCI dataset in data/UCI_datasets folder
    :param name: the name of the dataset
    :param sigma: sigma of the kernel to be applied
    """
    arr_feature = genfromtxt(f"data/UCI_datasets/{name}_features", delimiter=',')
    arr_labels = genfromtxt(f"data/UCI_datasets/{name}_labelled", delimiter=',', dtype=str)

    arr_labels = arr_labels[:,-1]

    unq_labels = np.unique(arr_labels)

    tot_dim = 0
    c_dims = []
    
    for label in unq_labels:
        len_label = len(np.where(arr_labels == label)[0]) 
        tot_dim += len_label 
        c_dims.append(len_label)
    
    NG = graph_from_points(arr_feature, sigma)

    GT = nbam.custom_noisy_matrix(tot_dim, c_dims, 0, 0)

    #plt.show(plt.imshow(NG))
    #plt.show(plt.imshow(GT))

    return NG, GT, tot_dim 

def get_GColi1_data():
    
    data = sp.loadmat("data/GCoil1.mat")
    NG = data['GCoil1']
    GT  = nbam.generate_matrix(72, 20, 0, 0, 'constant', 0)

    #plt.show(plt.imshow(NG))
    #plt.show(plt.imshow(GT))

    return NG, GT, 70*20


