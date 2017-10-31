from sklearn.cluster import KMeans
from sklearn import metrics
import scipy.io as sp
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/lakj/Documenti/university/thesis/code/dense_graph_reducer_forked/analysis/')
sys.path.insert(1, '/home/lakj/Documenti/university/thesis/code/dense_graph_reducer_forked/graph_reducer/')
import real_data as rd
import sensitivity_analysis as sa

def get_NG_t(name):
    data = sp.loadmat(f"/home/lakj/Documenti/university/thesis/code/dense_graph_reducer_forked/analysis/pymat/data/NG_ts/{name}_NG_t.mat")
    return data['NG_t']


#rounding problem ("column3C", 0.0004, 3),
dsets = [("ecoli", 0.475, 8), ("ionosphere", 0.12, 2), ("iris", 0.02, 3), ("userknowledge", 0.42, 4), ("spect-singleproton", 1.5, 2)]


for dset, sigma, k in dsets:

    print(f"---- {dset} {sigma} {k} ----")

    # 1. NG 
    NG, GT, tot_dim = rd.get_UCI_data(dset, sigma)
    # K-means 
    kmeans = KMeans(n_clusters=k).fit(GT)
    pred = kmeans.predict(NG)
    # Evaluation
    ars = metrics.adjusted_rand_score(kmeans.labels_, pred)
    mis = metrics.adjusted_mutual_info_score(kmeans.labels_, pred)
    print(f"NG : {ars:.4} {mis:.4}")


    # 2. Amplitude Commute Time Kernel
    NG_t = get_NG_t(dset)
    # K-means
    pred = kmeans.predict(abs(1-NG_t))
    # Evaluation
    ars = metrics.adjusted_rand_score(kmeans.labels_, pred)
    mis = metrics.adjusted_mutual_info_score(kmeans.labels_, pred)
    print(f"ACT: {ars:.4} {mis:.4}")


    # 3. SZE reconstructed
    SZE_rec = get_SZE_rec(NG)
    # K-means
    pred = kmeans.predict(SZE_rec)
    # Evaluation
    ars = metrics.adjusted_rand_score(kmeans.labels_, pred)
    mis = metrics.adjusted_mutual_info_score(kmeans.labels_, pred)
    print(f"SZE: {ars:.4} {mis:.4}")


