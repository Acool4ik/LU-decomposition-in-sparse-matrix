import numpy as np
import time
import scipy.io
import scipy.sparse as spsp
import matplotlib.pyplot as plt
import lu
import spectral_partitioning
import kernighan_lin

# calc time (f32)     2s            5s              24s
datasets = ["arc130.mtx", "494_bus.mtx", "dwt_1007.mtx"]
path_to_dataset = "./datasets/" + datasets[1]
dtype = np.float32
fseparate = kernighan_lin.kernighan_lin  # or spectral_partitioning.get_separate_perm
fseparate.method_name = "kernighan_lin"

if __name__ == '__main__':
    # read data (graph)
    A = spsp.csr_matrix(scipy.io.mmread(path_to_dataset), dtype=dtype)
    A = A + 5 * spsp.eye(A.shape[0])

    # gets decomposition
    start = time.time()
    P, L, U = lu.rec_partitioning(A, fseparate)
    end = time.time()
    print("time decomposition [s] = ", end - start)

    # checks accuracy
    LU = L.dot(U)
    PLUP = LU[P, :][:, P].copy()
    print("norm(A) = ", spsp.linalg.norm(A, ord='fro'))
    print("norm(LU) = ", spsp.linalg.norm(LU, ord='fro'))
    print("norm(P(LU)P - A) = ", spsp.linalg.norm(PLUP - A, ord='fro'))

    # shows plot (report)
    data = [A, LU, PLUP, L, U, PLUP - A]
    lab1 = "A, norm = " + str(np.round(spsp.linalg.norm(A, ord='fro'), 1))
    lab2 = "LU, norm = " + str(np.round(spsp.linalg.norm(LU, ord='fro'), 1))
    lab3 = "P(LU)P, norm = " + str(np.round(spsp.linalg.norm(PLUP, ord='fro'), 1))
    labels = [lab1, lab2, lab3, "L", "U", "P(LU)P - A"]
    fig, axs = plt.subplots(2, len(data) // 2)

    n = A.shape[0]
    subtitle = ("time decomposition [s] = " + str(np.round(end - start, 1)) + ", dataset: "
                + str(n) + "x" + str(n) + " vertexes, nnz = " + str(A.nnz) +
                ", method: " + str(fseparate.method_name))
    fig.suptitle(subtitle)

    for i in range(2):
        for j in range(len(data) // 2):
            axs[i, j].spy(data[i * 3 + j], aspect='equal', marker='.', markersize=4, precision=1e-5)
            axs[i, j].set_title(labels[i * 3 + j])
    plt.show()
