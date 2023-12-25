import numpy as np
import scipy.sparse as spsp


def lu_iter_(L, U, i, j, n, m):
    for k in range(j, j + m):
        a = U[k, k]
        for row in range(max(i, k + 1), i + n):
            b = U[row, k]
            if abs(b) > 1e-6:
                L.append((row, k, b / a))
                U[row] -= b * (U[k] / a)


def rec_partitioning_(P, L, U, i, n, fseparate):
    if n < 20: return  # extreme case

    P_local, a, b, g = fseparate(U[i:(i+n), i:(i+n)])  # gets permutation for block-arrow shape
    P_i = np.arange(0, U.shape[0], dtype=np.int32)  # perm in i-th iteration
    P_i[i:(i + n)] = P_local + i
    U[:] = U[P_i, :][:, P_i]  # apply permutation to U (inplace)
    P[:] = P[P_i]

    if max(a, b, g) < n:
        rec_partitioning_(P, L, U, i + a + b, g, fseparate)  # (gamma, gamma)
        rec_partitioning_(P, L, U, i, a, fseparate)  # (alpha, alpha)
        rec_partitioning_(P, L, U, i + a, b, fseparate)  # (beta, beta)


def rec_partitioning(A, fseparate, copy=True):
    P = np.arange(0, A.shape[0], dtype=np.int32)  # default perm
    L = [(i, i, 1) for i in range(0, A.shape[0])]  # fills diagonal by 1
    U = A.tolil(copy=copy)

    # P, L, U changes inplace
    rec_partitioning_(P, L, U, 0, A.shape[0], fseparate)
    P = np.argsort(P)  # applies transpose
    lu_iter_(L, U, 0, 0, A.shape[0], A.shape[0])

    # create sparse matrix from coo list
    row = [L[i][0] for i in range(len(L))]
    col = [L[i][1] for i in range(len(L))]
    data = [L[i][2] for i in range(len(L))]
    L = spsp.csr_matrix((data, (row, col)), dtype=A.dtype)

    return P, L, U
