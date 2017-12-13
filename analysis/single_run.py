from sensitivity_analysis import SensitivityAnalysis
import numpy as np
import putils as pu
import ipdb
import process_datasets as proc


n = 400
d = 0.70
dset_ids = [1,2,3,4]

for dset_id in dset_ids:
    ### 2. ###
    filename = f"./data/synthetic_graphs/{n}/{n}_{d:.2f}_{dset_id}.npz"

    data = proc.search_dset(filename, synth=True)

    ### 3. ###
    s = SensitivityAnalysis(data)


    bounds = s.find_bounds()

    if not s.bounds:
        np.savez_compressed(filename, G=data['G'], bounds=bounds)

    kec = s.find_partitions()

    if kec == {}:
        print(f"[x] {n}_{d:.2f}_{dset_id}.npy")

    else:

        ### 4. ###
        max_idx = -1
        max_k = -1

        for k in kec.keys():
            if kec[k][2] > max_idx:
                max_k = k
                max_idx = kec[k][2]

        #print(f"[+] Partition with the highest sze_idx k: {max_k} idx: {kec[max_k][2]:.4f}")

        ### 5. ###
        sze_rec = s.reconstruct_mat(0, kec[max_k][1], max_k)

        ### 6. ###
        dist = s.L2_metric(sze_rec)
        print(f"[OK] {n}_{d:.2f}_{dset_id}.npy l2d:{dist:.4f} sze_idx:{kec[max_k][2]:.4f}")
        with io.open(f'./data/synthetic_graphs/{n}.csv', 'a') as f:
            f.write(f"{n},{d:.2f},{dset_id},{dist:.4f},{kec[max_k][2]:.4f},{s.bounds[0]:.5f},{s.bounds[1]:.5f},1\n")

