"""
UTF-8
author: francesco-p
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

sys.path.insert(0, '/home/lakj/Documenti/university/thesis/code/dense_graph_reducer/misc') 
sys.path.insert(1, '/home/lakj/Documenti/university/thesis/code/dense_graph_reducer/graph_reducer') 
import szemeredi_lemma_builder as slb 
import noisyblockadjmat as nbam
import plot  

def L2_distance(GT, NG):
    """
    Compute the L2 distance between two matrices
    """
    diff = np.subtract(GT,NG)
    diff = np.square(diff)
    accu = diff.sum()

    return accu**(1.0/2.0)


def main():
    """
    Main function code
    """
    # Graph constants
    cluster_size = 1000
    n_clusters = 10
    intranoise_lvl = 0
    modality = "constant"
    noise_val = 0.5

    # Szemeredi algorithm constants
    kind = "alon"
    is_weighted = True
    random_initialization = True 
    random_refinement = False # No other choice
    drop_edges_between_irregular_pairs = True
    compression = 0.05
    
    # Intercluster noise percentage
    internoise_lvl = 0.5 

    # Fixed size clusters
    #graph  = nbam.generate_matrix(cluster_size, n_clusters, internoise_lvl, intranoise_lvl, modality, noise_val)    
    # OR
    # Custom size clusters
    c_dimensions = [5000, 2000, 2500, 500]
    graph  = nbam.custom_noisy_matrix(cluster_size*n_clusters, c_dimensions, internoise_lvl, noise_val)    
    
    # Needs to be defined inside the main scope because graph must be referenced outside, otherwise will use all the RAM since it's a recursive function
    def find_trivial_epsilon(epsilon1, epsilon2, k_min, tolerance):
        """
        Performs binary search to find the best trivial epsilon candidate: the first epsilon for which k=2
        del statements are essential to free unused memory
        """
        
        step = (epsilon2 - epsilon1)/2.0

        if step < tolerance:
            return epsilon2
        else:
            epsilon_middle = epsilon1 + step
            print(f"|{epsilon1:.6f}-----{epsilon_middle:.6f}------{epsilon2:.6f}|", end=" ") 
            srla = slb.generate_szemeredi_reg_lemma_implementation(kind, graph, epsilon_middle, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)
            regular, k, reduced_matrix = srla.run(iteration_by_iteration=False, verbose=False, compression_rate=compression)
            print(f"{k} {regular}")
            if regular:
                if k==k_min:
                    del srla
                    return find_trivial_epsilon(epsilon1, epsilon_middle, k_min, tolerance)
                if k>k_min: # could be an else  
                    del srla
                    return find_trivial_epsilon(epsilon_middle, epsilon2, k_min, tolerance) 
                else:
                    del srla
                    return -1 # WTF
            else:
                del srla
                return find_trivial_epsilon(epsilon_middle, epsilon2, k_min, tolerance)
   
    def find_edge_epsilon(epsilon1, epsilon2, tolerance):
        """
        Performs binary search to find the best edge epsilon candidate: the first epsilon for which the partition is regular 
        del statements are essential to free unused memory
        """
        
        # WE CAN DEVIDE BY 3 TO ADD A BIAS OR A FAST CONVERGENCE SINCE WE KNOW WHERE TO LOOK
        step = (epsilon2 - epsilon1)/2.0 

        if step < tolerance:
            return epsilon2
        else:
            epsilon_middle = epsilon1 + step
            print(f"|{epsilon1:.6f}-----{epsilon_middle:.6f}------{epsilon2:.6f}|", end=" ") 
            srla = slb.generate_szemeredi_reg_lemma_implementation(kind, graph, epsilon_middle, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)
            regular, k, reduced_matrix = srla.run(iteration_by_iteration=False, verbose=False, compression_rate=compression)
            print(f"{k} {regular}")
            if regular:
                del srla
                return find_edge_epsilon(epsilon1, epsilon_middle, tolerance)
            else:
                del srla
                return find_edge_epsilon(epsilon_middle, epsilon2, tolerance)


    print("Finding trivial epsilon...")
    epsilon2 = find_trivial_epsilon(0.1,0.9, 2, 0.0001)
    print("Trivial epsilon candidate: {0:.6f}".format(epsilon2))
    print("Finding edge epsilon...")
    epsilon1 = find_edge_epsilon(0.1,epsilon2,0.0001)
    print("Edge epsilon candidate: {0:.6f}".format(epsilon1))

    epsilons = [epsilon1]

    tries = 20.0
    offs = (epsilon2 - epsilon1) / tries 
    for i in range(1, int(tries)+1):
        epsilons.append(epsilon1 + (i*offs))
    #epsilons = [0.4228, 0.4338, 0.4559, 0.4669, 0.5001, 0.58847]

    k_dist = {} 
    thresholds = np.arange(0, 1.05, 0.05) 
            
    for epsilon in epsilons:

        srla = slb.generate_szemeredi_reg_lemma_implementation(kind, graph, epsilon, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)
        regular, k, reduced_matrix = srla.run(iteration_by_iteration=False, verbose=False, compression_rate=compression)

        print(f"{internoise_lvl} {epsilon:.6f} {k} {regular}")

        # If the partition is regular and we discovered a new k partition
        if (k not in k_dist) and regular:      
            # We keep track of the distances for each partition with k elements
            dists = []
            for thresh in thresholds: 
                reconstructed_graph = srla.reconstruct_original_mat(thresh)
                distance = L2_distance(graph, reconstructed_graph)
                dists.append(distance)
                print(distance)

            k_dist[k] = dists 
            
    aux = "_".join(str(e) for e in c_dimensions)
    name = f"./imgs/{internoise_lvl}_{aux}.npy"
    np.save(name , k_dist)

    for k in k_dist.keys():
        lab = f"k={k}"
        plt.plot(thresholds, k_dist[k], label=lab) 

    plt.title(f"Internoise level {internoise_lvl}")
    plt.ylabel('Distance')
    plt.xlabel('Reconstruction Threshold')
    plt.legend(loc='upper left')
    plt.savefig(f"./imgs/{internoise_lvl}_{aux}.png")

    #plt.show()
        

if __name__ == "__main__":

    main()
