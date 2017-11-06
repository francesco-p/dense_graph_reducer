"""
UTF-8
author: francesco-p

1. numpy array 32 bit
2. refactor reconstruction matrix

"""

import numpy as np
import matplotlib.pyplot as plt
#import ipdb

import sys
sys.path.insert(1, '../graph_reducer/')
import szemeredi_lemma_builder as slb
import noisyblockadjmat as nbam
import process_datasets as pd
import classes_pair as wcp



class SensitivityAnalysis:

    def __init__(self, dset, bounds=None):

        # NG GT
        self.set_dset(dset)

        # Find bounds parameters
        self.bounds = bounds
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
        self.dset = dset
        self.NG = self.dset['NG']
        self.GT = self.dset['GT']

    def run_alg(self, epsilon):
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
            print(f"|{epsilon1:.6f}-----{epsilon_middle:.6f}------{epsilon2:.6f}|", end=" ")
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
            print(f"|{epsilon1:.6f}-----{epsilon_middle:.6f}------{epsilon2:.6f}|", end=" ")
            regular, k, classes = self.run_alg(epsilon_middle) 
            print(f"{k} {regular}")
            if regular:
                del self.srla
                return self.find_edge_epsilon(epsilon1, epsilon_middle)
            else:
                del self.srla
                return self.find_edge_epsilon(epsilon_middle, epsilon2)


    def find_bounding_epsilons(self):

        if self.bounds:
            epsilon1 = self.bounds[0]
            epsilon2 = self.bounds[1]
        else:
            print("Finding trivial epsilon...")
            epsilon2 = self.find_trivial_epsilon(0, 1)
            print(f"Trivial epsilon candidate: {epsilon2:.6f}")
            print("Finding edge epsilon...")
            epsilon1 = self.find_edge_epsilon(0, epsilon2)
            print(f"Edge epsilon candidate: {epsilon1:.6f}")
        self.bounds = [epsilon1, epsilon2]
        self.epsilons = [epsilon1]
        # Try self.tries different epsilons inside the bounds
        offs = (epsilon2 - epsilon1) / self.tries
        for i in range(1, int(self.tries)+1):
            self.epsilons.append(epsilon1 + (i*offs))


    def find_partitions(self):
        self.k_e_c= {}
        for epsilon in self.epsilons:
            regular, k, classes = self.run_alg(epsilon)
            print(f"{epsilon:.6f} {k} {regular}")
            if (k not in self.k_e_c) and regular and k!=2:
                self.k_e_c[k] = (epsilon, classes)
        return self.k_e_c


    def thresholds_analysis(self, classes, k, measure):
        self.measures = []
        for thresh in self.thresholds:
            sze_rec = self.reconstruct_mat(thresh, classes, k)
            res = measure(self.GT, sze_rec)/(self.GT.shape[0])
            self.measures.append(res)
        return self.measures


    def reconstruct_mat(self, thresh, classes, k):
        reconstructed_mat = np.zeros((self.GT.shape[0], self.GT.shape[0]))
        for r in range(2, k + 1):
            r_nodes = classes == r
            for s in range(1, r):
                s_nodes = classes == s
                adj_mat = (self.NG > 0.0).astype(float)
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

    def save_data(self):
        # Save the bounds of the analysis
        with open("./imgs/bounds.txt", "a") as bounds_file:
            bounds_file.write(f"{title}_{epsilon1}_{epsilon2}\n")

        # Adds NOISY GRAPH line
        ng_dist = L2_distance(GT, NG)/tot_dim
        k_dist['NG'] = [ng_dist]*len(thresholds) 

        # Save a dictionary of the results (we can plot it directly with plot.py)
        name = f"./npy/{title}.npy"
        np.save(name , k_dist)

        # Generate the plot
        for k in k_dist.keys():
            lab = f"k={k}"
            plt.plot(thresholds, k_dist[k], label=lab)
        plt.title(f"{title}")
        plt.ylabel('Distance')
        plt.xlabel('Reconstruction Threshold')
        plt.legend(loc='lower right')
        plt.savefig(f"./imgs/{title}.png")

        # Controls --plot parameter
        if args.plot:
            plt.show()


