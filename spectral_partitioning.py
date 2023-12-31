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
    if eigval[1] < 1e-5:  # checks on not connected graph
        alpha = spsp.csgraph.depth_first_order(Laplacian, 0, return_predecessors=False)
        beta = np.setdiff1d(np.arange(0, len(colors), dtype=np.int32), alpha)
        gamma = np.array([], dtype=np.int32)
    else:
        gamma = set()
        for k in range(0, len(Laplacian.row)):
            i = Laplacian.row[k]
            j = Laplacian.col[k]
            if j > i and colors[i] != colors[j]:
                gamma.add(i)
                gamma.add(j)
        gamma = np.array(list(gamma), dtype=np.int32)
        beta = np.setdiff1d(np.where(colors > 0), gamma)
        alpha = np.setdiff1d(np.where(colors < 0), gamma)

    # creates permutation based on sets
    a, b, g = len(alpha), len(beta), len(gamma)
    P = np.concatenate((alpha, beta, gamma), dtype=np.int32)
    return P, a, b, g
