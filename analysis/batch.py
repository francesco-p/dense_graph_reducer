import matplotlib.pyplot as plt
import numpy as np
import process_datasets as proc
from sensitivity_analysis import SensitivityAnalysis



def L2_distance(GT, NG):
    """
    Compute the L2 distance between two matrices
    """
    diff = np.subtract(GT,NG)
    diff = np.square(diff)
    accu = diff.sum()

    return accu**(1.0/2.0)

NG, GT, tot_dim = proc.get_UCI_data('ecoli', 0.475)

data = {}
data['NG'] = NG
data['GT'] = GT

s = SensitivityAnalysis(data)

s.find_bounding_epsilons()
s.find_partitions()
kec = s.find_partitions()

for k in kec.keys():
    classes = kec[k][1]
    dists = s.thresholds_analysis(classes, k, L2_distance)
    print(dists)

    lab = f"k={k}"
    plt.plot(s.thresholds, dists, label=lab)
    plt.title("Ecoli Sigma 1315")
    plt.ylabel('Distance')
    plt.xlabel('Reconstruction Threshold')
    plt.legend(loc='lower right')


plt.show()








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
