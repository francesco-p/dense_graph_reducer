"""
UTF-8
author: francesco-p

1. numpy array 32 bit
2. refactor reconstruction matrix

"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.insert(1, '../graph_reducer/')
import szemeredi_lemma_builder as slb
import noisyblockadjmat as nbam
import process_datasets as pd


def L2_distance(GT, NG):
    """
    Compute the L2 distance between two matrices
    """
    diff = np.subtract(GT,NG)
    diff = np.square(diff)
    accu = diff.sum()

    return accu**(1.0/2.0)


def create_graphs(kind, args):
    
    if kind != 'real':
        internoise_lvl = args.internoise_lvl 
        intranoise_lvl = args.intranoise_lvl
        noise_val = 0
        if args.random_noise:
            modality = 'weighted'
        else:
            modality = 'constant'
            noise_val = args.constant_noise

        # Fixed
        if kind == 'fixed':

            cluster_size = args.cluster_size 
            n_clusters = args.n_clusters
            c_dimensions = [cluster_size, n_clusters]

            NG  = nbam.generate_matrix(cluster_size, n_clusters, internoise_lvl, intranoise_lvl, modality, noise_val)
            GT  = nbam.generate_matrix(cluster_size, n_clusters, 0, 0, 'constant', 0)

            title = f'in-{internoise_lvl}-c-'
            title += "x".join(str(e) for e in c_dimensions)

            return NG, GT, title, cluster_size*n_clusters

        # Custom
        else:

            c_dimensions = args.c_dimensions
            c_dimensions = list(map(int, c_dimensions)) 
            tot_dim = sum(c_dimensions) 

            # TODO intranoise -i not implemented
            NG  = nbam.custom_noisy_matrix(tot_dim, c_dimensions, internoise_lvl, noise_val)
            GT  = nbam.custom_noisy_matrix(tot_dim, c_dimensions, 0, 0)

            title = f'in-{internoise_lvl}-cs-'
            title += "-".join(str(e) for e in c_dimensions)
            return NG, GT, title, tot_dim 

    # Real
    else:
        dataset = args.dataset 
        if dataset == 'XPCA':
            sigma = args.sigma 
            to_remove = args.to_remove 
            NG, GT, tot_dim = pd.get_XPCA_data(sigma, to_remove)
            title = f'XPCA_dataset_{sigma:.3f}'
            if args.dryrun:
                plt.show(plt.imshow(NG))
                plt.show(plt.imshow(GT))
                sys.exit("Dryrun") # TODO
            return NG, GT, title, tot_dim

        elif dataset == 'GColi1':
            NG, GT, tot_dim = pd.get_GColi1_data()
            title = 'GColi1_dataset'
            return NG, GT, title, tot_dim

        elif dataset == 'UCI':
            sigma = args.sigma 
            name = args.UCI 
            NG, GT, tot_dim = pd.get_UCI_data(name, sigma)
            title = f'UCI_{name}_dataset_sigma_{sigma:.10f}'

            if args.dryrun:
                plt.show(plt.imshow(NG))
                plt.show(plt.imshow(GT))
                sys.exit("Dryrun") # TODO
            return NG, GT, title, tot_dim

def main(graph_type, args):
    """
    Main function code
    """
    # Szemeredi algorithm constants
    kind = "alon"
    is_weighted = True
    random_initialization = True
    random_refinement = False # No other choice
    drop_edges_between_irregular_pairs = True
    compression = 0.05

    NG, GT, title, tot_dim = create_graphs(graph_type, args) 

    # Needs to be defined inside the main scope because NG must be referenced outside, otherwise will use all the RAM since it's a recursive function
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
            srla = slb.generate_szemeredi_reg_lemma_implementation(kind, NG, epsilon_middle, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)
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
                    return -1 # WTF... just in case
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
            srla = slb.generate_szemeredi_reg_lemma_implementation(kind, NG, epsilon_middle, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)
            regular, k, reduced_matrix = srla.run(iteration_by_iteration=False, verbose=False, compression_rate=compression)
            print(f"{k} {regular}")
            if regular:
                del srla
                return find_edge_epsilon(epsilon1, epsilon_middle, tolerance)
            else:
                del srla
                return find_edge_epsilon(epsilon_middle, epsilon2, tolerance)

    # If we specified the bounds use them otherwise search them
    if args.bounds:
            bounds = list(map(float, args.bounds)) 
            epsilon1 = bounds[0]
            epsilon2 = bounds[1]
    else:
        print("Finding trivial epsilon...")
        epsilon2 = find_trivial_epsilon(0.1,0.9, 2, 0.00001)
        print("Trivial epsilon candidate: {0:.6f}".format(epsilon2))
        print("Finding edge epsilon...")
        epsilon1 = find_edge_epsilon(0.1,epsilon2,0.00001)
        print("Edge epsilon candidate: {0:.6f}".format(epsilon1))

    epsilons = [epsilon1]

    # Try 20 different epsilons inside the bounds
    tries = 20.0
    offs = (epsilon2 - epsilon1) / tries
    for i in range(1, int(tries)+1):
        epsilons.append(epsilon1 + (i*offs))

    # Dictrionary to hold all the different distances of each different partition k
    k_dist = {}
    thresholds = np.arange(0, 1.05, 0.05)

    # For each epsilon compute szemeredi and if we have never seen a partition of k elements then reduce the noisy graph and compute the error
    for epsilon in epsilons:

        srla = slb.generate_szemeredi_reg_lemma_implementation(kind, NG, epsilon, is_weighted, random_initialization, random_refinement, drop_edges_between_irregular_pairs)
        regular, k, reduced_matrix = srla.run(iteration_by_iteration=False, verbose=False, compression_rate=compression)

        print(f"{title} {epsilon:.6f} {k} {regular}")

        # If the partition is regular and we discovered a new k partition
        if (k not in k_dist) and regular and k!=2:
            # We keep track of the distances for each partition with k elements
            dists = []
            for thresh in thresholds:
                reconstructed_graph = srla.reconstruct_original_mat(thresh)
                #distance = L2_distance(NG, reconstructed_graph) # Studies compression capacities
                distance = L2_distance(GT, reconstructed_graph)/tot_dim # Studies noise filtering
                dists.append(distance)
                print(distance)

            k_dist[k] = dists


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis of Szemeredi algorithm")

    subparsers = parser.add_subparsers(title='Graph generation', description='Modality of the creation of the graphs', help='You must select one of between the three possibilities')

    # Fixed graph generation 
    fixed_p = subparsers.add_parser("fixed", help="Fixed generation of clusters")
    fixed_p.add_argument("-bounds", "-b", help="Two epsilon bound", nargs=2, type=float)
    fixed_p.add_argument("-cluster_size", "-s", help="Size of a single square cluster", type=int, default=1000)
    fixed_p.add_argument("-n_clusters", "-n", help="Number of clusters", type=int, default=10)
    fixed_p.add_argument("-internoise_lvl", "-e", help="Percentage of noise between the clusters (0 for no noise) the noise weight is a Uniform(0,1) distribution", type=float, default=0.5)
    fixed_p.add_argument("-intranoise_lvl", "-i", help="Percentage of noise within each cluster (0 for no noise) the noise weight is a Uniform(0,1) distribution", type=float, default=0)
    fixed_p.add_argument("--plot", help="Show a plot of the generated matrix", action="store_true")
    group = fixed_p.add_mutually_exclusive_group(required=True)
    group.add_argument("-constant_noise", help="Set constant nature of noise", type=float, default=0.5)
    group.add_argument("-random_noise", help="Set uniform nature of noise (between 0 and 1)", action="store_true")


    # Custom graph generation 
    custom_p = subparsers.add_parser("custom", help="Custom generation of imbalanced clusters")
    custom_p.add_argument("-bounds", "-b", help="Two epsilon bound", nargs=2, type=float)
    custom_p.add_argument("-internoise_lvl", "-e", help="Percentage of noise between the clusters (0 for no noise) the noise weight is a Uniform(0,1) distribution", type=float, default=0.5)
    custom_p.add_argument("-intranoise_lvl", "-i", help="Percentage of noise within each cluster (0 for no noise) the noise weight is a Uniform(0,1) distribution", type=float, default=0)
    custom_p.add_argument("c_dimensions", help="Cluster dimensions", nargs='*')
    custom_p.add_argument("--plot", help="Show a plot of the generated matrix", action="store_true")
    group1 = custom_p.add_mutually_exclusive_group(required=True)
    group1.add_argument("-constant_noise", help="Set constant nature of noise", type=float, default=0.5)
    group1.add_argument("-random_noise", help="Set uniform nature of noise (between 0 and 1)", action="store_true")

    # Real dataset
    real_p = subparsers.add_parser("real", help="Graph from real dataset")
    real_p.add_argument("-bounds", "-b", help="Two epsilon bound", nargs=2, type=float)
    real_p.add_argument("dataset", help="Dataset name", choices=["XPCA", "GColi1", "UCI"])
    real_p.add_argument("-UCI", "-u", help="Dataset from UCI requires a name")
    real_p.add_argument("-sigma", "-s", help="Dataset sigma", type=float, default=0.0124)
    real_p.add_argument("-dryrun",  help="Performs a dry run then exits", action="store_true")

    # Only with xpca
    real_p.add_argument("-to_remove", "-r", help="Dataset number of columns to remove", type=int, default=0)

    real_p.add_argument("--plot", help="Show a plot of the generated matrix", action="store_true")

    args = parser.parse_args()


    main(sys.argv[1], args)
