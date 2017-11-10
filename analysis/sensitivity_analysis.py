"""
UTF-8
author: francesco-p

1. numpy array 32 bit
2. refactor reconstruction matrix

"""

import numpy as np
import matplotlib.pyplot as plt
import process_datasets as pd
from sklearn import metrics
import sys
sys.path.insert(1, '../graph_reducer/')
import szemeredi_lemma_builder as slb



class SensitivityAnalysis:

    def __init__(self, dset):

        # NG GT
        self.set_dset(dset)

        # Find bounds parameters
        self.min_k = 2
        self.min_step = 0.00001
        self.tries = 20
        self.thresholds = np.arange(0, 1.05, 0.05)


        # SZE algorithm parameters
        self.kind = "alon"
        self.is_weighted = True
        self.random_initialization = True
        self.random_refinement = False # No other choice
        self.drop_edges_between_irregular_pairs = True


        # SZE running parameters
        self.iteration_by_iteration = False
        self.verbose = False
        self.compression = 0.05


    def set_dset(self, dset):
        """
        Change dataset
        :param dset: the new dictionary hoding NG GT and the bounds
        """
        self.dset = dset
        self.NG = self.dset['NG']
        self.GT = self.dset['GT']
        self.labels = self.dset['labels']
        self.bounds = list(self.dset['bounds']) # to pass the test in find bounding epsilons


    def run_alg(self, epsilon):
        """
        Creates and run the szemeredi algorithm with a particular dataset
        epsilon: the epsilon parameter of the algorithm
        return: if the partition found is regular, its cardinality, and how the nodes are partitioned
        """
        self.srla = slb.generate_szemeredi_reg_lemma_implementation(self.kind,
                                                                    self.NG,
                                                                    epsilon,
                                                                    self.is_weighted,
                                                                    self.random_initialization,
                                                                    self.random_refinement,
                                                                    self.drop_edges_between_irregular_pairs)
        regular, k, classes = self.srla.run(iteration_by_iteration=self.iteration_by_iteration,
                                                    verbose=self.verbose,
                                                    compression_rate=self.compression)
        return regular, k, classes


    def find_trivial_epsilon(self, epsilon1, epsilon2):
        """
        Performs binary search to find the best trivial epsilon candidate: the first epsilon for which k=2
        del statements are essential to free unused memory
        """
        step = (epsilon2 - epsilon1)/2.0
        if step < self.min_step:
            return epsilon2
        else:
            epsilon_middle = epsilon1 + step
            print(f"    |{epsilon1:.6f}-----{epsilon_middle:.6f}------{epsilon2:.6f}|", end=" ")
            regular, k, classes = self.run_alg(epsilon_middle)
            print(f"{k} {regular}")

            if regular:
                if k==self.min_k:
                    del self.srla
                    return self.find_trivial_epsilon(epsilon1, epsilon_middle)
                if k>self.min_k: # could be an else
                    del self.srla
                    return self.find_trivial_epsilon(epsilon_middle, epsilon2)
                else:
                    del self.srla
                    return -1 # WTF... just in case
            else:
                del self.srla
                return self.find_trivial_epsilon(epsilon_middle, epsilon2)

    def find_edge_epsilon(self, epsilon1, epsilon2):
        step = (epsilon2 - epsilon1)/2.0
        if step < self.min_step:
            return epsilon2
        else:
            epsilon_middle = epsilon1 + step
            print(f"    |{epsilon1:.6f}-----{epsilon_middle:.6f}------{epsilon2:.6f}|", end=" ")
            regular, k, classes = self.run_alg(epsilon_middle)
            print(f"{k} {regular}")
            if regular:
                del self.srla
                return self.find_edge_epsilon(epsilon1, epsilon_middle)
            else:
                del self.srla
                return self.find_edge_epsilon(epsilon_middle, epsilon2)


    def find_bounding_epsilons(self):
        """
        Finds the bounding epsilons and set up the range where to search
        """
        if self.bounds:
            epsilon1 = self.bounds[0]
            epsilon2 = self.bounds[1]
        else:
            print("     Finding trivial epsilon...")
            epsilon2 = self.find_trivial_epsilon(0, 1)
            print(f"    Trivial epsilon candidate: {epsilon2:.6f}")
            print("    Finding edge epsilon...")
            epsilon1 = self.find_edge_epsilon(0, epsilon2)
            print(f"    Edge epsilon candidate: {epsilon1:.6f}")
        self.bounds = [epsilon1, epsilon2]
        self.epsilons = [epsilon1]
        # Try self.tries different epsilons inside the bounds
        offs = (epsilon2 - epsilon1) / self.tries
        for i in range(1, int(self.tries)+1):
            self.epsilons.append(epsilon1 + (i*offs))


    def find_partitions(self):
        """
        Start looking for partitions
        return: a dictionary with key the cardinality of the partition, the corresponding epsilon and the classes reduced array
        """
        self.k_e_c= {}
        for epsilon in self.epsilons:
            regular, k, classes = self.run_alg(epsilon)
            print(f"    {epsilon:.6f} {k} {regular}")
            if (k not in self.k_e_c) and regular and k!=2:
                self.k_e_c[k] = (epsilon, classes)
        return self.k_e_c


    def thresholds_analysis(self, classes, k, measure):
        """
        Start performing threshold analysis with a given measure
        :param classes: the reduced array
        :param k: the cardinality of the patition
        :param measure: the measure to use
        :return: the measures
        """
        self.measures = []
        for thresh in self.thresholds:
            sze_rec = self.reconstruct_mat(thresh, classes, k)
            res = measure(sze_rec)
            print(f"    {res:.5f}")
            self.measures.append(res)
        return self.measures


    def reconstruct_mat(self, thresh, classes, k):
        """
        Reconstruct the original matrix from a reduced one.
        :param thres: the edge threshold if the density between two pairs is over it we put an edge
        :param classes: the reduced graph expressed as an array
        :return: a numpy matrix of the size of GT
        """
        reconstructed_mat = np.zeros((self.GT.shape[0], self.GT.shape[0]), dtype='float32')
        # TODO Is it always a complete graph????
        adj_mat = (self.NG > 0.0).astype('float32')
        for r in range(2, k + 1):
            r_nodes = classes == r
            for s in range(1, r):
                s_nodes = classes == s
                index_map = np.where(classes == r)[0]
                index_map = np.vstack((index_map, np.where(classes == s)[0]))
                bip_sim_mat = self.NG[np.ix_(index_map[0], index_map[1])]
                bip_adj_mat = adj_mat[np.ix_(index_map[0], index_map[1])]
                n = bip_sim_mat.shape[0]
                bip_density = bip_sim_mat.sum() / (n ** 2.0)
                # Put edges if above threshold
                if bip_density > thresh:
                    reconstructed_mat[np.ix_(r_nodes, s_nodes)] = reconstructed_mat[np.ix_(s_nodes, r_nodes)] = bip_density
        np.fill_diagonal(reconstructed_mat, 0.0)
        return reconstructed_mat


    #################
    #### Metrics ####
    #################

    def knn_voting_system(self, graph):
        """
        Implements knn voting system to calculate if the labeling is correct.
        :param graph: graph
        :returns: adjusted random score
        """
        n = len(self.labels) 
        k = 9
        candidates = np.zeros(n, dtype='int16')
        i = 0
        for row in graph:
            max_k_idxs = row.argsort()[-k:]
            aux = row[max_k_idxs] > 0
            k_indices = max_k_idxs[aux]

            if len(k_indices) == 0:
                k_indices = row.argsort()[-1:]

            #candidate_lbl = np.bincount(self.labels[k_indices].astype(int)).argmax()
            candidate_lbl = np.bincount(self.labels[k_indices]).argmax()
            candidates[i] = candidate_lbl
            i += 1

        ars = metrics.adjusted_rand_score(self.labels, candidates)

        return ars


    def L2_distance(self, graph):
        """
        Compute the L2 distance between two matrices
        np.linalg.norm(a-b)
        """
        diff = np.subtract(self.GT, graph)
        diff = np.square(diff)
        accu = diff.sum()

        return accu**(1.0/2.0)



