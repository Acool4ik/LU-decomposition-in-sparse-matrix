import numpy as np
import scipy.sparse as spsp


def lu_iter_(L, U, i, j, n, m):
    for k in range(j, j + m):
        a = U[k, k]
        for row in range(max(i, k + 1), i + n):
            b = U[row, k]
            if b != 0:
                L.append((row, k, b / a))
                U[row] -= b * (U[k] / a)


def rec_partitioning_(P, L, U, i, n, fseparate, trace_iters, is_debug):
    if n < 10 or trace_iters[0] >= trace_iters[1]:  # extreme case
        if is_debug:
            print("extreme case, n = ", n)
        lu_iter_(L, U, i, i, n, n)
        return

    P_local, a, b, g = fseparate(U[i:(i+n), i:(i+n)])  # gets permutation for block-arrow shape
    P_i = np.arange(0, U.shape[0], dtype=np.int32)  # perm in i-th iteration
    P_i[i:(i + n)] = P_local + i
    U[:] = U[P_i, :][:, P_i]  # apply permutation to U (inplace)
    P[:] = P[P_i]
    trace_iters[0] += 1

    if is_debug:
        print("A,B,G = ", a, b, g)

    if max(a, b, g) == n:  # inf recursion case (when next divide impossible)
        if is_debug:
            print("extreme case G = A u B")
        lu_iter_(L, U, i, i, n, n)
    else:
        rec_partitioning_(P, L, U, i, a, fseparate, trace_iters, is_debug)  # (alpha, alpha)
        rec_partitioning_(P, L, U, i + a, b, fseparate, trace_iters, is_debug)  # (beta, beta)
        lu_iter_(L, U, i + a + b, i, g, a)  # lu over (gamma,alpha) part
        lu_iter_(L, U, i + a + b, i + a, g, b)  # lu over (gamma,beta) part
        rec_partitioning_(P, L, U, i + a + b, g, fseparate, trace_iters, is_debug)  # (gamma, gamma)


def rec_partitioning(A, fseparate, copy=True, max_iters=7, is_debug=False):
    P = np.arange(0, A.shape[0], dtype=np.int32)  # default perm
    L = [(i, i, 1) for i in range(0, A.shape[0])]  # fills diagonal by 1
    U = A.tolil(copy=copy)

    # P, L, U changes inplace
    trace_iters = [1, max_iters]  # current iter, max iterations
    rec_partitioning_(P, L, U, 0, A.shape[0], fseparate, trace_iters, is_debug)
    P = np.argsort(P)  # applies transpose

    if is_debug:
        print("cnt iters = ", trace_iters[0])

    # create sparse matrix from coo list
    row = [L[i][0] for i in range(len(L))]
    col = [L[i][1] for i in range(len(L))]
    data = [L[i][2] for i in range(len(L))]
    L = spsp.csr_matrix((data, (row, col)), dtype=A.dtype)

    return P, L, U
