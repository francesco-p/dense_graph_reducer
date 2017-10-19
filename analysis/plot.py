"""
Usage: python plot.py dictionary.npy internoise_lvl
"""
import numpy as np
import matplotlib.pyplot as plt
import sys


def main(filename, internoise_lvl):
    """
    Requires a dictionary dict[k] = list(errors)
    """

    dictionary = np.load(filename).item()

    thresholds = np.arange(0, 1, 0.05)

    for k in dictionary.keys():
        lab = "k={0}".format(k)
        plt.plot(thresholds, dictionary[k], label=lab) 

    plt.title("Internoise level {0}".format(internoise_lvl))
    plt.ylabel('Distance')
    plt.xlabel('Reconstruction Threshold')
    plt.legend(loc='upper left')

    plt.show()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
