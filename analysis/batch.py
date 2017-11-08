import matplotlib.pyplot as plt
import numpy as np
import os
import ipdb
import process_datasets as proc
from sensitivity_analysis import SensitivityAnalysis


def search_dset(name, sigma):
    path = "data/npz/"
    for f in os.listdir(path):
        if f == f"{name}_{sigma:.5f}.npz":
            return np.load(path+f) 
    return None



def L2_distance(GT, NG):
    """
    Compute the L2 distance between two matrices
    """
    diff = np.subtract(GT,NG)
    diff = np.square(diff)
    accu = diff.sum()

    return accu**(1.0/2.0)


# Retrieve data
#NG, GT, n = proc.get_data('ecoli', 0.475)
name = 'ecoli'
sigma = 0.475

print("[+] Searching for existing .npz")
data = search_dset(name, sigma)
first_time = False
if not data:
    first_time = True
    print("No data found: Generating data")

    if name == 'XPCA':
        NG, GT, n = proc.get_XPCA_data(sigma, 0)
    else:
        NG, GT, n = proc.get_data(name, sigma)

    data = {}
    data['NG'] = NG
    data['GT'] = GT
    data['bounds'] = []
else:
    print("[+] Found .npz")

# Start analysis
s = SensitivityAnalysis(data)

print("[+] Finding bounding epsilon")
s.find_bounding_epsilons()

if first_time:
    np.savez(f"data/npz/{name}_{sigma:.5f}.npz", NG=NG, GT=GT, bounds=s.bounds)
    print("[+] Data saved")

print("[+] Finding partitions")
kec = s.find_partitions()
print("[+] Performing threshold analysis")
for k in kec.keys():
    print(f"  k: {k}")
    classes = kec[k][1]
    dists = s.thresholds_analysis(classes, k, L2_distance)

    lab = f"k={k}"
    plt.plot(s.thresholds, dists, label=lab)
    plt.title(f"{name}_{sigma:.5f}")
    plt.ylabel('Distance')
    plt.xlabel('Reconstruction Threshold')
    plt.legend(loc='lower right')


plt.show()






#def create_graphs(kind, args):
#
#    if kind != 'real':
#        internoise_lvl = args.internoise_lvl
#        intranoise_lvl = args.intranoise_lvl
#        noise_val = 0
#        if args.random_noise:
#            modality = 'weighted'
#        else:
#            modality = 'constant'
#            noise_val = args.constant_noise
#
#        # Fixed
#        if kind == 'fixed':
#
#            cluster_size = args.cluster_size
#            n_clusters = args.n_clusters
#            c_dimensions = [cluster_size, n_clusters]
#
#            NG  = nbam.generate_matrix(cluster_size, n_clusters, internoise_lvl, intranoise_lvl, modality, noise_val)
#            GT  = nbam.generate_matrix(cluster_size, n_clusters, 0, 0, 'constant', 0)
#
#            title = f'in-{internoise_lvl}-c-'
#            title += "x".join(str(e) for e in c_dimensions)
#
#            return NG, GT, title, cluster_size*n_clusters
#
#        # Custom
#        else:
#
#            c_dimensions = args.c_dimensions
#            c_dimensions = list(map(int, c_dimensions))
#            tot_dim = sum(c_dimensions)
#
#            # TODO intranoise -i not implemented
#            NG  = nbam.custom_noisy_matrix(tot_dim, c_dimensions, internoise_lvl, noise_val)
#            GT  = nbam.custom_noisy_matrix(tot_dim, c_dimensions, 0, 0)
#
#            title = f'in-{internoise_lvl}-cs-'
#            title += "-".join(str(e) for e in c_dimensions)
#            return NG, GT, title, tot_dim
#
#    # Real
#    else:
#        dataset = args.dataset
#        if dataset == 'XPCA':
#            sigma = args.sigma
#            to_remove = args.to_remove
#            NG, GT, tot_dim = pd.get_XPCA_data(sigma, to_remove)
#            title = f'XPCA_dataset_{sigma:.3f}'
#            if args.dryrun:
#                plt.show(plt.imshow(NG))
#                plt.show(plt.imshow(GT))
#                sys.exit("Dryrun") # TODO
#            return NG, GT, title, tot_dim
#
#        elif dataset == 'GColi1':
#            NG, GT, tot_dim = pd.get_GColi1_data()
#            title = 'GColi1_dataset'
#            return NG, GT, title, tot_dim
#
#        elif dataset == 'UCI':
#            sigma = args.sigma
#            name = args.UCI
#            NG, GT, tot_dim = get_UCI_data(name, sigma)
#            title = f'UCI_{name}_dataset_sigma_{sigma:.10f}'
#
#            if args.dryrun:
#                plt.show(plt.imshow(NG))
#                plt.show(plt.imshow(GT))
#                sys.exit("Dryrun") # TODO
#            return NG, GT, title, tot_dim


#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Analysis of Szemeredi algorithm")
#
#    subparsers = parser.add_subparsers(title='Graph generation', description='Modality of the creation of the graphs', help='You must select one of between the three possibilities')
#
#    # Fixed graph generation
#    fixed_p = subparsers.add_parser("fixed", help="Fixed generation of clusters")
#    fixed_p.add_argument("-bounds", "-b", help="Two epsilon bound", nargs=2, type=float)
#    fixed_p.add_argument("-cluster_size", "-s", help="Size of a single square cluster", type=int, default=1000)
#    fixed_p.add_argument("-n_clusters", "-n", help="Number of clusters", type=int, default=10)
#    fixed_p.add_argument("-internoise_lvl", "-e", help="Percentage of noise between the clusters (0 for no noise) the noise weight is a Uniform(0,1) distribution", type=float, default=0.5)
#    fixed_p.add_argument("-intranoise_lvl", "-i", help="Percentage of noise within each cluster (0 for no noise) the noise weight is a Uniform(0,1) distribution", type=float, default=0)
#    fixed_p.add_argument("--plot", help="Show a plot of the generated matrix", action="store_true")
#    group = fixed_p.add_mutually_exclusive_group(required=True)
#    group.add_argument("-constant_noise", help="Set constant nature of noise", type=float, default=0.5)
#    group.add_argument("-random_noise", help="Set uniform nature of noise (between 0 and 1)", action="store_true")
#
#
#    # Custom graph generation
#    custom_p = subparsers.add_parser("custom", help="Custom generation of imbalanced clusters")
#    custom_p.add_argument("-bounds", "-b", help="Two epsilon bound", nargs=2, type=float)
#    custom_p.add_argument("-internoise_lvl", "-e", help="Percentage of noise between the clusters (0 for no noise) the noise weight is a Uniform(0,1) distribution", type=float, default=0.5)
#    custom_p.add_argument("-intranoise_lvl", "-i", help="Percentage of noise within each cluster (0 for no noise) the noise weight is a Uniform(0,1) distribution", type=float, default=0)
#    custom_p.add_argument("c_dimensions", help="Cluster dimensions", nargs='*')
#    custom_p.add_argument("--plot", help="Show a plot of the generated matrix", action="store_true")
#    group1 = custom_p.add_mutually_exclusive_group(required=True)
#    group1.add_argument("-constant_noise", help="Set constant nature of noise", type=float, default=0.5)
#    group1.add_argument("-random_noise", help="Set uniform nature of noise (between 0 and 1)", action="store_true")
#
#    # Real dataset
#    real_p = subparsers.add_parser("real", help="Graph from real dataset")
#    real_p.add_argument("-bounds", "-b", help="Two epsilon bound", nargs=2, type=float)
#    real_p.add_argument("dataset", help="Dataset name", choices=["XPCA", "GColi1", "UCI"])
#    real_p.add_argument("-UCI", "-u", help="Dataset from UCI requires a name")
#    real_p.add_argument("-sigma", "-s", help="Dataset sigma", type=float, default=0.0124)
#    real_p.add_argument("-dryrun",  help="Performs a dry run then exits", action="store_true")
#
#    # Only with xpca
#    real_p.add_argument("-to_remove", "-r", help="Dataset number of columns to remove", type=int, default=0)
#
#    real_p.add_argument("--plot", help="Show a plot of the generated matrix", action="store_true")
#
#    args = parser.parse_args()
#
#
#    main(sys.argv[1], args)
