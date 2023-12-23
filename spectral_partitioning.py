import numpy as np
import scipy.sparse as spsp
import networkx as nx


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
    P = np.zeros(len(colors), dtype=np.int32)
    start = 0
    P[alpha] = np.arange(start, start + len(alpha))
    start += len(alpha)
    if b > 0: P[beta] = np.arange(start, start + len(beta))
    start += len(beta)
    if g > 0: P[gamma] = np.arange(start, start + len(gamma))
    P = P.argsort()
    return P, a, b, g


def print_(A, a, b):
    print("alpha,alpha = ", spsp.linalg.norm(A[:a, :a], ord='fro'))
    print("beta,beta = ", spsp.linalg.norm(A[a:(a + b), a:(a + b)], ord='fro'))
    print("gamma,gamma = ", spsp.linalg.norm(A[(a + b):, (a + b):], ord='fro'))
    print("alpha,beta = ", spsp.linalg.norm(A[:a, a:(a + b)], ord='fro'))
    print("beta,alpha = ", spsp.linalg.norm(A[a:(a + b), :a], ord='fro'))


if __name__ == '__main__':
    # case 1: connected graph
    G = nx.read_gml('karate.gml')
    A = nx.to_scipy_sparse_array(G)
    P, a, b, g = get_separate_perm(A)
    A = (A[P, :])[:, P]  # apply perm on column and rows
    print_(A, a, b)

    # case 2: not connected graph
    A = spsp.csr_matrix([
        [1,1,0,1,0,0],
        [1,1,0,1,0,0],
        [0,0,1,0,1,1],
        [1,1,0,1,0,0],
        [0,0,1,0,1,1],
        [0,0,1,0,1,1]
    ])
    P, a, b, g = get_separate_perm(A)
    A = (A[P, :])[:, P]
    print_(A, a, b)
