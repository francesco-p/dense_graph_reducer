"""
UTF-8
author: francesco-p
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/lakj/Documenti/university/thesis/code/dense_graph_reducer/misc') 
sys.path.insert(1, '/home/lakj/Documenti/university/thesis/code/dense_graph_reducer/graph_reducer') 

import szemeredi_lemma_builder as slb 
import noisyblockadjmat as nbam

def compute_distance(GT, NG):
    """
    Compute the L2 distance between two matrices
    """
    diff = np.subtract(GT,NG)
    diff = np.square(diff)
    accu = diff.sum()
    return accu**(1/2)

def main():
    """
    Main function code
    """
    # Adj Mat constants
    cluster_size = 1000 #Should be 1000
    n_clusters = 10  #Should be 10
    intranoise_lvl = 0 
    modality = "constant" 
    noise_val = 0.5

    # Szemeredi algorithm constants
    kind = "alon"
    is_weighted = True
    random_initialization = True 
    random_refinement = False # No choice random:TODO
    drop_edges_between_irregular_pairs = True
    compression = 0.05
    
    
    # To analyze
    internoise_lvl = 0.5 
    #epsilons = [0.6756, 0.6825, 0.68280, 0.686]

    epsilons = []
    epsilon1 = 0.681
    epsilon2 = 0.68758
    tries = 100.0
    offs = (epsilon2 - epsilon1) / tries 
    for i in range(1, int(tries)+1):
        epsilons.append(epsilon1 + (i*offs))

    sim_mat  = nbam.generate_matrix(cluster_size, n_clusters, internoise_lvl, intranoise_lvl, modality, noise_val)    
    #sim_mat  = nbam.custom_noisy_matrix(cluster_size*n_clusters, [7000,3000], internoise_lvl, noise_val)    

    k_dist = {} 
            
    for epsilon in epsilons:

        print("------\n{0}_{1:.6f}".format(internoise_lvl, epsilon)) 
                
        srla = slb.generate_szemeredi_reg_lemma_implementation(kind, sim_mat, epsilon, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)

        regular, k, reduced_matrix = srla.run(iteration_by_iteration=False, verbose=True, compression_rate=compression)

        if (k not in k_dist) and regular:      
            dists = []
            #name = "./imgs/red_{0}_{1:.6f}".format(internoise_lvl, epsilon)
            #np.save(name, reduced_matrix)
            for thresh in np.arange(0, 1, 0.05): 
                ng = srla.reconstruct_original_mat(thresh, 0)
                #name = "./imgs/rec_{0}_{1:.6f}".format(internoise_lvl, epsilon)
                #np.save(name , ng)
                dist = compute_distance(sim_mat, ng)
                dists.append(dist)
                #name = "./imgs/{0}_{1:.6f}_{2}.npy".format(internoise_lvl, epsilon, k)
                #np.save(name , dists)
                print(dist)

            k_dist[k] = dists 
            
    name = "./imgs/{0}_all2.npy".format(internoise_lvl)
    np.save(name , k_dist)

        

if __name__ == "__main__":
    main()
