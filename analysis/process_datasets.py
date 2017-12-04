import scipy.io as sp
import numpy as np
import ipdb
import pandas as pd



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
    w_graph = np.zeros((n,n), dtype='float32')

    for i in range(0,n):
        copy = np.tile(np.array(x[i, :]), (i+1, 1))
        difference = copy - x[0:i+1, :]
        column = np.exp(-sigma*(difference**2).sum(1))

        #w_graph[0:i+1, i] = column
        w_graph[0:i, i] = column[:-1] # set diagonal to 0 the resulting graph is different

    return w_graph + w_graph.T


def get_data(name, sigma):
    """
    Given a .csv features:label it returns the dataset modified with a gaussian kernel
    :param name: the name of the dataset,it must be in the data folder
    :param sigma: sigma of the gaussian kernel
    :return: NG, GT, labels
    """

    df = pd.read_csv(f"data/{name}.csv", delimiter=',', header=None)
    labels = df.iloc[:,-1].astype('category').cat.codes.values
    features = df.values[:,:-1].astype('float32')

    unq_labels, unq_counts = np.unique(labels, return_counts=True)

    NG = graph_from_points(features, sigma)
    aux, GT, aux2 = custom_cluster_matrix(len(labels), unq_counts, 0, 0)

    return NG.astype('float32'), GT.astype('int32'), labels


def get_GCoil1_data():
    data = sp.loadmat("data/GCoil1.mat")
    NG = data['GCoil1']
    GT  = cluster_matrix(72, 20, 0, 0, 'constant', 0)
    return NG.astype('float32'), GT.astype('int32'), np.repeat(np.array([x for x in range(0,20)]), 72)


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
    aux, GT, labels = custom_cluster_matrix(tot_dim, c_dimensions, 0, 0)

    return NG, GT, labels


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


def custom_cluster_matrix(mat_dim, dims, internoise_lvl, noise_val):
    """
    Custom noisy matrix
    :param mat_dim : dimension of the whole graph
    :param dims: list of cluster dimensions
    :param internoise_lvl : level of noise between clusters
    :param noise_lvl : value of the noise
    :returns: NG, GT, labels
    """
    if len(dims) > mat_dim:
        sys.exit("You want more cluster than nodes???")
        return 0

    if sum(dims) != mat_dim:
        sys.exit("The sum of clusters dimensions must be equal to the total number of nodes")
        return 0

    # TODO does not implement intranoise
    mat = np.tril(np.random.random((mat_dim, mat_dim)) < internoise_lvl, -1)
    mat = np.multiply(mat, noise_val)
    x = 0
    for dim in dims:
        mat[x:x+dim,x:x+dim]= np.tril(np.ones(dim), -1)
        x += dim

    m = (mat + mat.T).astype('float32')
    return m, m.astype('int16'), np.repeat(range(1, len(dims)+1,), dims)


def cluster_matrix(cluster_size, n_clusters, internoise_lvl, intranoise_lvl, modality, noise_val):
    """
    Generate a noisy adjacency matrix with noisy cluster over the diagonal. The computed matrix will have size = n_cluster * cluster_size

    :param n_clusters: number of cluster
    :param cluster_size: size of a single cluster
    :param internoise_lvl: percentage of noise between the clusters (0.0 for no noise)
    :param intranoise_lvl: percentage of noise within a cluster (0.0 for completely connected clusters)
    :param modality: the nature of the noise. Currently the supported values are 'weighted' and 'constant'
    :param noise_val: the constant value to represent noise, used in combination with mode='constant'

    :return: the noisy block adjacency matrix
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

