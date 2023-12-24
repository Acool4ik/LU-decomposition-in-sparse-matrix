import numpy as np
import time
import scipy.io
import networkx as nx
import scipy.sparse as spsp
import matplotlib.pyplot as plt
import lu
import spectral_partitioning

path_to_dataset = "./datasets/494_bus.mtx"
#path_to_dataset = "./datasets/karate.gml"

if __name__ == '__main__':
    # read data (graph)
    A = spsp.csr_matrix(scipy.io.mmread(path_to_dataset))
    #A = nx.to_scipy_sparse_array(nx.read_gml(path_to_dataset))
    #A = A + 5 * spsp.eye(A.shape[0])

    # gets decomposition
    start = time.time()
    P, L, U = lu.rec_partitioning(A, spectral_partitioning.get_separate_perm)
    end = time.time()
    print("time decomposition [s] = ", end - start)

    # checks accuracy
    LU = L.dot(U)
    print("norm(A) = ", spsp.linalg.norm(A, ord='fro'))
    print("norm(LU) = ", spsp.linalg.norm(LU, ord='fro'))
    print("norm(P(LU)P - A) = ", spsp.linalg.norm(LU[P, :][:, P] - A, ord='fro'))

    # shows plot (report)
    data = [A, LU, L, U]
    fig, axs = plt.subplots(1, len(data))
    for i in range(len(data)):
        axs[i].spy(data[i], aspect='equal', marker='.', markersize=2, precision=1e-5)
    plt.show()
