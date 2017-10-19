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
    sim_mat  = nbam.generate_matrix(cluster_size, n_clusters, internoise_lvl, intranoise_lvl, modality, noise_val)    
    # OR
    # Custom size clusters
    #sim_mat  = nbam.custom_noisy_matrix(cluster_size*n_clusters, [7000,3000], internoise_lvl, noise_val)    
    
    # Needs to be defined inside the main scope because sim_mat must be referenced outside, otherwise will use all the RAM since it's a recursive function
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
            print("|{0:.6}-----{1:.6}------{2:.6}|".format(epsilon1, epsilon_middle, epsilon2), end=" ") 
            srla = slb.generate_szemeredi_reg_lemma_implementation(kind, sim_mat, epsilon_middle, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)
            regular, k, reduced_matrix = srla.run(iteration_by_iteration=False, verbose=False, compression_rate=compression)
            if regular:
                if k==k_min:
                    del srla
                    return find_trivial_epsilon(epsilon1, epsilon_middle, k_min, tolerance)
                if k>k_min: # could be an else  
                    del srla
                    return find_trivial_epsilon(epsilon_middle, epsilon2, k_min, tolerance) 
                else:
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
            return epsilon1
        else:
            epsilon_middle = epsilon1 + step
            print("|{0:.6}-----{1:.6}------{2:.6}|".format(epsilon1, epsilon_middle, epsilon2), end=" ") 
            srla = slb.generate_szemeredi_reg_lemma_implementation(kind, sim_mat, epsilon_middle, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)
            regular, k, reduced_matrix = srla.run(iteration_by_iteration=False, verbose=False, compression_rate=compression)
            if regular:
                del srla
                return find_edge_epsilon(epsilon1, epsilon_middle, tolerance)
            else:
                del srla
                return find_edge_epsilon(epsilon_middle, epsilon2, tolerance)


    trivial = find_trivial_epsilon(0.3,0.9, 2, 0.0001)
    print("Trivial epsilon candidate: {0:.6f}".format(trivial))
    edge = find_edge_epsilon(0.5,trivial,0.0001)
    print("Edge epsilon candidate: {0:.6f}".format(edge))

    
    # Fixed
    #epsilons = [0.6756, 0.6825, 0.68280, 0.686]
    # OR
    # Dynamic
    epsilons = []
    epsilon1 = edge 
    epsilon2 = trivial 
    tries = 20.0
    offs = (epsilon2 - epsilon1) / tries 
    for i in range(1, int(tries)+1):
        epsilons.append(epsilon1 + (i*offs))

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
                dist = L2_distance(sim_mat, ng)
                dists.append(dist)
                #name = "./imgs/{0}_{1:.6f}_{2}.npy".format(internoise_lvl, epsilon, k)
                #np.save(name , dists)
                print(dist)

            k_dist[k] = dists 
            
    #name = "./imgs/{0}_all2.npy".format(internoise_lvl)
    #np.save(name , k_dist)

    for k in k_dist.keys():
        lab = "k={0}".format(k)
        plt.plot(np.arange(0, 1, 0.05), k_dist[k], label=lab) 

    plt.title("Internoise level {0}".format(internoise_lvl))
    plt.ylabel('Distance')
    plt.xlabel('Reconstruction Threshold')
    plt.legend(loc='upper left')

    plt.show()
        

if __name__ == "__main__":
    main()
