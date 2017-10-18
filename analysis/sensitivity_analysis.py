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
    Compute the distance between two matrices
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

    # Szemeredi algorithm constants
    kind = "alon"
    is_weighted = True
    random_initialization = True 
    random_refinement = False # No choice random:TODO
    drop_edges_between_irregular_pairs = True
    
    
    modality = "constant" 
    noise_vals = [0.5]

    compression = 0.05
    
    internoise_lvls = [0.5]
    
    epsilons = [0.68285]

    for noise_val in noise_vals:

        for internoise_lvl in internoise_lvls:

            sim_mat  = nbam.generate_matrix(cluster_size, n_clusters, internoise_lvl, intranoise_lvl, modality, noise_val)    
            
            for epsilon in epsilons:

                print("------\n{0}_{1}".format(internoise_lvl, epsilon)) 
                
                srla = slb.generate_szemeredi_reg_lemma_implementation(kind, sim_mat, epsilon, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)

                success, reduced_matrix = srla.run(iteration_by_iteration=False, verbose=True, compression_rate=compression)

                if not success:
                    print("Failure")
                else:
                    print("Success")    
                    name = "./imgs/{0}_{1}".format(internoise_lvl, epsilon) 
                    np.save(name, reduced_matrix)
                    #plt.imshow(reduced_matrix)
                    #plt.show()
                    #for thresh in [0.5, 1]:
                    #ng = srla.reconstruct_original_mat(0.5)
                    #plt.imshow(ng)
                    #plt.show()
                    #print(compute_distance(sim_mat, ng))
                        # TODO

        

if __name__ == "__main__":
    main()
