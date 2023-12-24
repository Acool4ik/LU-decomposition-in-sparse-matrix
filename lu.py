import numpy as np
import scipy.sparse as spsp
import networkx as nx
import matplotlib.pyplot as plt


def lu_iter_(L, U, i, j, n, m):
    for k in range(j, j + m):
        a = U.getrow(k).getcol(k).toarray()[0][0]
        for row in range(max(i, k + 1), i + n):
            b = U.getrow(row).getcol(k).toarray()[0][0]
            if b != 0:
                L.append((row, k, b / a))
                U[row] = U.getrow(row) - b * (U.getrow(k) / a)


def rec_partitioning_(P, L, U, i, n, fseparate):
    if n < 10:  # extreme case
        lu_iter_(L, U, i, i, n, n)
        return

    P_local, a, b, g = fseparate(U[i:(i+n), i:(i+n)])  # gets permutation for block-arrow shape
    P_i = np.arange(0, U.shape[0])  # perm in i-th iteration
    P_i[i:(i+n)] = P_local + i
    U[:] = U[P_i, :][:, P_i]  # apply permutation to U (inplace)
    P[:] = P[P_i]

    if max(a, b, g) == n:  # inf recursion case (when next divide impossible)
        lu_iter_(L, U, i, i, n, n)
    else:
        rec_partitioning_(P, L, U, i, a, fseparate)  # (alpha, alpha)
        lu_iter_(L, U, i + a + b, i, g, a)  # lu over (gamma,alpha) part
        rec_partitioning_(P, L, U, i + a, b, fseparate)  # (beta, beta)
        lu_iter_(L, U, i + a + b, i + a, g, b)  # lu over (gamma,beta) part
        rec_partitioning_(P, L, U, i + a + b, g, fseparate)  # (gamma, gamma)


def rec_partitioning(A, fseparate, copy=True):
    P = np.arange(0, A.shape[0])  # default perm
    L = [(i, i, 1) for i in range(0, A.shape[0])]  # fills diagonal by 1

    if copy: U = A.copy()
    else: U = A

    # P, L, U changes inplace
    rec_partitioning_(P, L, U, 0, A.shape[0], fseparate)

    P = np.argsort(P)  # applies transpose

    # create sparse matrix from coo list
    row = [L[i][0] for i in range(len(L))]
    col = [L[i][1] for i in range(len(L))]
    data = [L[i][2] for i in range(len(L))]
    L = spsp.coo_matrix((data, (row, col)))

    return P, L, U
