import matplotlib.pyplot as plt 
import numpy as np
import networkx as nx
import ipdb

def density(adj_mat, indices_a, indices_b):
    """ Calculates the density between two sets of vertices
    :param indices_a: the indices of the first set
    :param indices_b: the indices of the second set
    """
    if np.array_equal(indices_a, indices_b):
        n = len(indices_a)
        max_edges = (n*(n-1))/2
        n_edges = np.tril(adj_mat[np.ix_(indices_a, indices_a)], -1).sum()
        return n_edges / max_edges

    n_a = len(indices_a)
    n_b = len(indices_b)
    max_edges = n_a * n_b
    n_edges = adj_mat[np.ix_(indices_a, indices_b)].sum()
    return n_edges / max_edges


def test_density():
    fullycon = np.ones((5,5))
    bin_tree = nx.to_numpy_array(nx.balanced_tree(2,3))

    indices_a = np.array([0,1,3])
    indices_b = np.array([0,1,3])
    assert density(bin_tree, indices_a, indices_a) == 2/3

    indices_a = np.array([0,3,9])
    indices_b = np.array([0,3,9])
    assert density(bin_tree, indices_a, indices_b) == 0

    indices_a = np.array([0,2,4])
    indices_b = np.array([0,2,4])
    assert density(fullycon, indices_a, indices_b) == 1

    indices_a = np.array([7,8])
    indices_b = np.array([9,10])
    assert density(bin_tree, indices_a, indices_b) == 0

    indices_a = np.array([3,8])
    indices_b = np.array([1,4])
    assert density(bin_tree, indices_a, indices_b) == 1/4


def compute_indensities(k, classes, mat):
    """ Compute the inside density for each class of a given partition
    :returns: np.array(float32) of densities for each class in the partition
    """
    cls = list(range(0, k + 1))
    densities = np.zeros(len(cls), dtype='float32')
    for c in cls:
        c_indices = np.where(classes == c)[0]
        if c_indices.size:
            densities[c] = density(mat, c_indices, c_indices)
        else:
            densities[c] = 0

    return densities

def test_compute_indensities():
    n = 3
    ar = np.array([1,0])
    two_clust = np.vstack((np.tile(np.repeat(ar, n), (n,1)), np.tile(np.repeat(ar[::-1]*2, n), (n,1))))
    k = 2

    classes = np.array([1,1,1,2,2,2])

    two_clust = np.vstack((np.tile(np.repeat(ar, n), (n,1)), np.tile(np.repeat(ar[::-1], n), (n,1))))
    assert np.array_equal(np.array([0,1,1]), compute_indensities(k, classes, two_clust))


def fill_new_set(self, new_set, compls, maximize_density):
    """ Find nodes that can be added
    Move from compls the nodes in can_be_added until we either finish the nodes or reach the desired cardinality
    """

    if maximize_density:
        nodes = self.adj_mat(np.ix_(new_set, compls)) == 1.0
    else:
        nodes = self.adj_mat(np.ix_(new_set, compls)) == 0.0

    # These are the nodes that can be added to certs
    # [TODO] I can exploit duplicates since they denote more connected nodes!
    # The ideal thing is to order them by the number of occurrences
    to_add = np.unique(np.tile(compls, (len(new_set), 1))[nodes])

    while new_set.size < self.classes_cardinality:

        # If there are nodes in to_add, we keep moving from compls to new_set
        if to_add.size > 0:
            node, to_add = to_add[-1], to_add[:-1]
            new_set.append(node)
            compls = np.delete(compls, np.argwhere(compls == node))

        else:
            # If there aren't candidate nodes, we keep moving from complements 
            # to certs until we reach the desired cardinality
            node, compls = compls[-1], compls[:-1]
            new_set.append(node)


def indeg_guided(self):
    """
    Refinement:
        Dati i certificates, calcolo la loro densità interna. Se è sopra una certa soglia, li splitto
        rispettiamente in due assegnandoli così:
                h1          h2
                h3          h4
                h5 ...

        h1 > h2 > h3 ... dove h1,h2,h3 sono gli hub di un certificate.
        Aumento i set nuovi cercando di preservare la indensity

        Se densità interna è minore della threshold allora campiono uniformemente i certificate e aggiungo
        nodi dall'unione dei complements cercando di tenere la densità bassa
    """

    threshold = 0.7

    to_be_refined = list(range(1, self.k + 1))
    is_classes_cardinality_odd = self.classes_cardinality % 2 == 1
    self.classes_cardinality //= 2

    in_densities = compute_indensities()

    new_k = 0

    while to_be_refined:
        s = to_be_refined.pop(0)
        irregular_r_indices = []

        for r in to_be_refined:
            if self.certs_compls_list[r - 2][s - 1][0][0]:
                irregular_r_indices.append(r)

        # If class s has irregular classes
        if irregular_r_indices:

            # Choose candidate based on the inside-outside density index
            r = choose_candidate(self, in_densities, s, irregular_r_indices)
            to_be_refined.remove(r)

            s_certs = self.certs_compls_list[r - 2][s - 1][0][1]
            r_certs = self.certs_compls_list[r - 2][s - 1][0][0]
            s_compls = self.certs_compls_list[r - 2][s - 1][1][1]
            r_compls = self.certs_compls_list[r - 2][s - 1][1][0]

            # Merging the two complements
            compls = np.append(s_compls, r_compls)

            # Calculating certificates densities
            dens_s_cert = density(self, s_cert, s_cert)
            dens_r_cert = density(self, r_cert, r_cert)

            for cert, dens in [(s_cert, dens_s_cert), (r_cert, dens_r_cert)]:

                # Indices of the cert ordered by in-degree, it doesn't matter if we reverse the list as long as we unzip it 
                #degs = self.adj_mat[np.ix_(cert, cert)].sum(1).argsort()[::-1]
                degs = self.adj_mat[np.ix_(cert, cert)].sum(1).argsort()

                # Density of s is above the threshold, we augment density
                if dens > threshold:

                    # Unzip them in half to preserve seeds
                    set1=  cert[degs[0:][::2]]
                    set2 =  cert[degs[1:][::2]]

                    # Adjust cardinality of the new set
                    fill_new_set(self, set1, compls, True)
                    self.classes[set1] = k + 1

                    fill_new_set(self, set2, compls, True)
                    self.classes[set2] = k + 2

                    new_k -= 1
                    self.classes[set1] = new_k
                    new_k -= 1
                    self.classes[set2] = new_k

                else:
                    # Low density branch
                    set1 = np.random.choice(cert, len(degs)//2)
                    set2 = np.setdiff1d(cert, set1)

                    fill_new_set(self, set1, compls, False)
                    fill_new_set(self, set2, compls, False)

                    new_k -= 1
                    self.classes[set1] = new_k
                    new_k -= 1
                    self.classes[set2] = new_k

        else:
            # The class is e-reg with all the others then we split structure
            s_indegs = within_degrees(self, s)

            set1=  cert[s_indegs[0:][::2]]
            set2=  cert[s_indegs[1:][::2]]

            new_k -= 1
            self.classes[set1] = new_k
            new_k -= 1
            self.classes[set2] = new_k

    self.classes *= -1
    self.k *= 2

    return True


