"""
UTF-8
author: francesco-p
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp


def synthetic_regular_partition(k, epsilon):
    """
    Generates a synthetic regular partition.
    :param k: the cardinality of the partition
    :param epsilon: the epsilon parameter to calculate the number of irregular pairs
    :return: a weighted symmetric graph
    """

    # Generate a kxk matrix where each element is between (0,1]
    mat = np.tril(1-np.random.random((k, k)), -1)

    x = np.tril_indices_from(mat, -1)[0]
    y = np.tril_indices_from(mat, -1)[1]

    # Generate a random number between 0 and epsilon*k**2 (number of irregular pairs)
    n_irr_pairs = round(np.random.uniform(0, epsilon*(k**2)))

    # Select the indices of the irregular  pairs
    irr_pairs = np.random.choice(len(x), n_irr_pairs)

    mat[(x[irr_pairs],  y[irr_pairs])] = 0

    return mat + mat.T

def graph_from_points(x, sigma):
    """
    Generates a graph (weighted graph) from a set of points x (nxd) and a sigma decay
    :param x: a numpy matrix of n times d dimension
    :param sigma: a sigma for the gaussian kernel
    :return: a weighted symmetric graph
    """

    n = x.shape[0]
    w_graph = np.zeros((n,n))

    for i in range(0,n):
        copy = np.tile(np.array(x[i, :]), (i+1, 1))
        difference = copy - x[0:i+1, :]
        column = np.exp(-sigma*(difference**2).sum(1))

        #w_graph[0:i+1, i] = column
        w_graph[0:i, i] = column[:-1] # set diagonal to 0 the resulting graph is different

    return w_graph + w_graph.T


def get_real_noisy_data():
    """
    Generates a graph from the noisy mat
    """

    data = sp.loadmat("/home/lakj/Documenti/university/thesis/code/dense_graph_reducer_forked/analysis/data/XPCA.mat")
    m = data['X']

    return graph_from_points(m, 0.0248)


def custom_noisy_matrix(mat_dim, dims, internoise_lvl, noise_val):
    """
    Custom noisy matrix
    mat_dim : dimension of the whole graph
    dims: list of cluster dimensions
    internoise_lvl : level of noise between clusters
    noise_lvl : value of the noise
    """
    if len(dims) > mat_dim:
        sys.exit("You want more cluster than nodes???")
        return 0

    if sum(dims) != mat_dim:
        sys.exit("The sum of clusters dimensions must be equal to the total number of nodes")
        return 0

    mat = np.tril(np.random.random((mat_dim, mat_dim)) < internoise_lvl, -1)
    mat = np.multiply(mat, noise_val)
    x = 0
    for dim in dims:
        mat[x:x+dim,x:x+dim]= np.tril(np.ones(dim), -1)
        x += dim

    return mat + mat.T

def generate_matrix(cluster_size, n_clusters, internoise_lvl, intranoise_lvl, modality, noise_val):
    """
    Generate a noisy adjacency matrix with noisy cluster over the diagonal. The computed matrix will have size = n_cluster * cluster_size

    n_clusters: number of cluster
    cluster_size: size of a single cluster
    internoise_lvl: percentage of noise between the clusters (0.0 for no noise)
    intranoise_lvl: percentage of noise within a cluster (0.0 for completely connected clusters)
    modality: the nature of the noise. Currently the supported values are 'weighted' and 'constant'
    noise_val: the constant value to represent noise, used in combination with mode='constant'

    Returns the noisy block adjacency matrix
    """

    mat_size = cluster_size * n_clusters

    # Interclass noise
    if internoise_lvl != 0:
        mat = np.tril(np.random.random((mat_size, mat_size)) < internoise_lvl, -1)
        if modality == "weighted":
            mat = np.multiply(mat, np.random.random((mat_size, mat_size)))
        elif modality == "constant":
            mat = np.multiply(mat, noise_val)
        else:
            sys.exit("incorrect modality")
    else:
        mat = np.zeros((mat_size, mat_size))

    for i in range(1, n_clusters+1):
        # Intraclass noise
        cl = np.tril(np.ones((cluster_size, cluster_size)), -1)

        if intranoise_lvl != 0:
            noise = np.tril(np.random.random((cluster_size, cluster_size)) < intranoise_lvl, -1)

            if modality == "weighted":
                noise = np.multiply(noise, np.random.random((cluster_size, cluster_size)))
            elif modality == "constant":
                noise = np.multiply(noise, 1.0 - noise_val)
            else:
                sys.exit("incorrect modality")

            cl = cl - noise

        mat[((i - 1) * cluster_size):(i * cluster_size), ((i - 1) * cluster_size):(i * cluster_size)] = cl

    mat = mat + mat.T

    return mat

def main():
    """
    Main function code
    """
    parser = argparse.ArgumentParser(description="""Generate a noisy adjacency matrix with noisy cluster over the diagonal. The computed matrix will have size = n_cluster * cluster_size""")
    parser.add_argument("cluster_size", help="Size of a single square cluster", type=int)
    parser.add_argument("n_cluster", help="Number of clusters", type=int)
    parser.add_argument("internoise_lvl", help="Percentage of noise between the clusters (0 for no noise) the noise weight is a Uniform(0,1) distribution", type=float)
    parser.add_argument("intranoise_lvl", help="Percentage of noise within each cluster (0 for no noise) the noise weight is a Uniform(0,1) distribution", type=float)
    parser.add_argument("--constant", help="Set constant nature of noise, if this parameter is omitted weighted noise is assumed (an edge has random number from 0 to 1)", type=float)

    parser.add_argument("--plot", help="Show a plot of the generated matrix", action="store_true")


    args = parser.parse_args()

    cluster_size = args.cluster_size
    n_clusters = args.n_cluster
    internoise_lvl = args.internoise_lvl
    intranoise_lvl = args.intranoise_lvl

    if args.constant:
        modality = "constant"
        noise_val = args.constant
    else:
        modality = "weighted"
        noise_val = 0

    mat = generate_matrix(cluster_size, n_clusters, internoise_lvl, intranoise_lvl, modality, noise_val)

    if args.plot:
        plt.imshow(mat)
        plt.show()

    return mat

if __name__ == '__main__':
    main()

