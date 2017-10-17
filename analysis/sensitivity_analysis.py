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
    cluster_size = 1000
    n_clusters = 10 
    intranoise_lvl = 0 
    modality = "constant" 
    noise_val = 1

    # Szemeredi algorithm constants
    kind = "alon"
    is_weighted = True
    random_initialization = True 
    random_refinement = False # No choice random:TODO
    drop_edges_between_irregular_pairs = True


    internoise_lvl = 0.5  # Analysis
    
    sim_mat  = nbam.generate_matrix(cluster_size, n_clusters, internoise_lvl, intranoise_lvl, modality, noise_val)    
    
    plt.imshow(sim_mat)
    plt.show()

    for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:

        print("{0}\nepsilon: {1} internoise_lvl:{2}".format("-"*6, epsilon, internoise_lvl))

        srla = slb.generate_szemeredi_reg_lemma_implementation(kind, sim_mat, epsilon, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)

        reduced_matrix = srla.run(iteration_by_iteration=False, verbose=True, compression_rate=0.05)

        if reduced_matrix.sum() == 0 :
            print("Failure")
        else:
            print("Success")    
            plt.imshow(reduced_matrix)
            plt.show()
            #for thresh in [0.5, 1]:
            #ng = srla.reconstruct_original_mat(0.5)
            #plt.imshow(ng)
            #plt.show()
            #print(compute_distance(sim_mat, ng))
                # TODO

    

if __name__ == "__main__":
    main()
