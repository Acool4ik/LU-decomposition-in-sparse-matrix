import numpy as np
import scipy.sparse as spsp
import networkx as nx
import matplotlib.pyplot as plt


def lu_iter(A, i, j, n):
    return A


def get_separate_perm(mat):
    # finds Fiedler vector
    Laplacian = spsp.csgraph.laplacian(mat, dtype=np.float32)
    eigval, eigvec = spsp.linalg.eigsh(Laplacian, k=2, which="SM")
    colors = np.sign(eigvec[:, 1])

    # gets 3 sets, where gamma - separator
    if eigval[1] < 1e-8:  # checks on not connected graph
        alpha = spsp.csgraph.depth_first_order(Laplacian, 0, return_predecessors=False)
        beta = np.setdiff1d(np.arange(0, len(colors)), alpha)
        gamma = np.array([])
    else:
        gamma = set()
        for k in range(0, len(Laplacian.row)):
            i = Laplacian.row[k]
            j = Laplacian.col[k]
            if j > i and colors[i] != colors[j]:
                gamma.add(i)
                gamma.add(j)
        gamma = np.array(list(gamma))
        beta = np.setdiff1d(np.where(colors > 0), gamma)
        alpha = np.setdiff1d(np.where(colors < 0), gamma)

    # creates permutation based on sets
    a, b, g = len(alpha), len(beta), len(gamma)
    start = 0
    P = np.zeros(len(colors), dtype=np.int32)
    if a > 0: P[alpha] = np.arange(start, start + a)
    start += a
    if b > 0: P[beta] = np.arange(start, start + b)
    start += b
    if g > 0: P[gamma] = np.arange(start, start + g)
    P = P.argsort()
    return P, a, b, g


def rec_partitioning(A, i, n):
    if n < 3:
        return lu_iter(A, i, i, n)

    P_local, a, b, g = get_separate_perm(A[i:(i+n), i:(i+n)])
    P = np.arange(0, A.shape[0])
    P[i:(i+n)] = P_local + i
    A = A[P, :][:, P]

    if a < n:
        A = rec_partitioning(A, i, a)
        A = lu_iter(A, i + a + b, i, g)
    if b < n:
        A = rec_partitioning(A, i + a, b)
        A = lu_iter(A, i + a + b, i + a, g)
    if g < n:
        A = rec_partitioning(A, i + a + b, g)
    return A


if __name__ == '__main__':
    A = nx.to_scipy_sparse_array(nx.read_gml('karate.gml'))
    A = A + spsp.eye(A.shape[0])
    A = rec_partitioning(A, 0, A.shape[0])
    plt.spy(A, aspect='equal', marker='.', markersize=5)
    plt.grid(True)
    plt.xticks(np.arange(0, A.shape[0], step=2))
    plt.yticks(np.arange(0, A.shape[0], step=2))
    plt.show()
