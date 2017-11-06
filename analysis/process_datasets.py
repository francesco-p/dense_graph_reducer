import scipy.io as sp
import math
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/lakj/Documenti/university/thesis/code/dense_graph_reducer_forked/misc')
import noisyblockadjmat as nbam

def create_graphs(kind, args):
    
    if kind != 'real':
        internoise_lvl = args.internoise_lvl 
        intranoise_lvl = args.intranoise_lvl
        noise_val = 0
        if args.random_noise:
            modality = 'weighted'
        else:
            modality = 'constant'
            noise_val = args.constant_noise

        # Fixed
        if kind == 'fixed':

            cluster_size = args.cluster_size 
            n_clusters = args.n_clusters
            c_dimensions = [cluster_size, n_clusters]

            NG  = nbam.generate_matrix(cluster_size, n_clusters, internoise_lvl, intranoise_lvl, modality, noise_val)
            GT  = nbam.generate_matrix(cluster_size, n_clusters, 0, 0, 'constant', 0)

            title = f'in-{internoise_lvl}-c-'
            title += "x".join(str(e) for e in c_dimensions)

            return NG, GT, title, cluster_size*n_clusters

        # Custom
        else:

            c_dimensions = args.c_dimensions
            c_dimensions = list(map(int, c_dimensions)) 
            tot_dim = sum(c_dimensions) 

            # TODO intranoise -i not implemented
            NG  = nbam.custom_noisy_matrix(tot_dim, c_dimensions, internoise_lvl, noise_val)
            GT  = nbam.custom_noisy_matrix(tot_dim, c_dimensions, 0, 0)

            title = f'in-{internoise_lvl}-cs-'
            title += "-".join(str(e) for e in c_dimensions)
            return NG, GT, title, tot_dim 

    # Real
    else:
        dataset = args.dataset 
        if dataset == 'XPCA':
            sigma = args.sigma 
            to_remove = args.to_remove 
            NG, GT, tot_dim = pd.get_XPCA_data(sigma, to_remove)
            title = f'XPCA_dataset_{sigma:.3f}'
            if args.dryrun:
                plt.show(plt.imshow(NG))
                plt.show(plt.imshow(GT))
                sys.exit("Dryrun") # TODO
            return NG, GT, title, tot_dim

        elif dataset == 'GColi1':
            NG, GT, tot_dim = pd.get_GColi1_data()
            title = 'GColi1_dataset'
            return NG, GT, title, tot_dim

        elif dataset == 'UCI':
            sigma = args.sigma 
            name = args.UCI 
            NG, GT, tot_dim = get_UCI_data(name, sigma)
            title = f'UCI_{name}_dataset_sigma_{sigma:.10f}'

            if args.dryrun:
                plt.show(plt.imshow(NG))
                plt.show(plt.imshow(GT))
                sys.exit("Dryrun") # TODO
            return NG, GT, title, tot_dim

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


