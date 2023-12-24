import numpy as np
import scipy.sparse as spsp
import networkx as nx
import matplotlib.pyplot as plt
import lu
import spectral_partitioning


if __name__ == '__main__':
    # read data (graph)
    A = nx.to_scipy_sparse_array(nx.read_gml('karate.gml'))
    A = A + 5 * spsp.eye(A.shape[0])

    # gets decomposition
    P, L, U = lu.rec_partitioning(A, spectral_partitioning.get_separate_perm)
    LU = L.dot(U)

    # checks accuracy
    print("norm(A) = ", spsp.linalg.norm(A, ord='fro'))
    print("norm(LU) = ", spsp.linalg.norm(LU, ord='fro'))
    print("norm(P(LU)P - A) = ", spsp.linalg.norm(LU[P, :][:, P] - A, ord='fro'))

    # shows plot (report)
    fig, axs = plt.subplots(1, 4)
    data = [A, LU, L, U]
    for i in range(len(data)):
        axs[i].spy(data[i], aspect='equal', marker='.', markersize=5)
        axs[i].grid(True)
        axs[i].xaxis.set_ticks(np.arange(0, A.shape[0], step=2))
        axs[i].yaxis.set_ticks(np.arange(0, A.shape[0], step=2))
    plt.show()
