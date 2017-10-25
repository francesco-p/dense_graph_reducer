"""
Usage: python plot.py dictionary.npy <title of the plot>
"""
import numpy as np
import matplotlib.pyplot as plt
import sys


def main(filename, title):
    """
    Requires a dictionary dict[k] = list(errors)
    """

    dictionary = np.load(filename).item()

    thresholds = np.arange(0, 1.05, 0.05)

    for k in dictionary.keys():
        lab = "k={0}".format(k)
        plt.plot(thresholds, dictionary[k], label=lab) 

    s = ""
    for word in title:
        s += " " + word
    #plt.title("Internoise level {0}".format(internoise_lvl))
    plt.title(s)
    plt.ylabel('Distance')
    plt.xlabel('Reconstruction Threshold')
    plt.legend(loc='upper right')

    plt.show()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
