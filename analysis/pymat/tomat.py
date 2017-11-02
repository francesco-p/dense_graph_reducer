import sys
sys.path.insert(0, '/home/lakj/Documenti/university/thesis/code/dense_graph_reducer_forked/analysis/')

import process_datasets as pd
import scipy.io as spio
import matplotlib.pyplot as plt

#worst = ["indian-liver", "pop-failures", "spect-singleproton", "spect-test"] 


dsets = [("column3C", 0.0004), ("ecoli", 0.475), ("ionosphere", 0.12), ("iris", 0.02), ("userknowledge", 0.42), ("spect-singleproton", 1.5)]


for dset, sigma in dsets:

    data = {}

    print(f"Processing dataset: {dset} ...")

    NG, GT, tot_dim = pd.get_UCI_data(dset, sigma)

    data['NG'] = NG
    data['GT'] = GT.astype(float)
    data['sigma'] = sigma 

    spio.savemat(f"data/{dset}.mat", data)


    plt.imshow(NG)
    plt.savefig(f"data/pngs/{dset}_{sigma}_NG.png")
    plt.imshow(GT)
    plt.savefig(f"data/pngs/{dset}_{sigma}_GT.png")

    print("Saved")



#import matlab.engine
#eng = matlab.engine.start_matlab()
