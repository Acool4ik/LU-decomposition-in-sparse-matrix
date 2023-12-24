import numpy as np
import scipy.sparse as spsp
import networkx as nx


def get_separate_perm(mat):
    # finds Fiedler vector
    G = nx.Graph(mat)
    mat = nx.adjacency_matrix(G, weight=None)
    G = nx.Graph(mat)
    Laplacian = nx.laplacian_matrix(G).asfptype().tocoo()  # fix excessive copying!

    eigval, eigvec = spsp.linalg.eigsh(Laplacian, k=2, which="SM")
    colors = np.sign(eigvec[:, 1])

    # gets 3 sets, where gamma - separator
    if eigval[1] < 1e-6:  # checks on not connected graph
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
