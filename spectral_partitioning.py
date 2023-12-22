import numpy as np
import scipy.sparse as spsp
import networkx as nx


def get_separate_perm(mat):
    # finds Fiedler vector
    Laplacian = spsp.csgraph.laplacian(mat, dtype=np.float32)
    eigval, eigvec = spsp.linalg.eigsh(Laplacian, k=2, which="SM")
    colors = np.sign(eigvec[:, 1])

    if eigval[1] < 1e-8:  # checks on not connected graph
        return None

    # gets 3 sets, where gamma - separator
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
    a, b, g = len(alpha), len(beta), len(gamma)

    # creates permutation based on sets
    P = np.zeros(len(colors), dtype=np.int32)
    start = 0
    P[alpha] = np.arange(start, start + len(alpha))
    start += len(alpha)
    P[beta] = np.arange(start, start + len(beta))
    start += len(beta)
    P[gamma] = np.arange(start, start + len(gamma))
    P = P.argsort()
    return P, a, b, g

G = nx.read_gml('karate.gml')
A = spsp.csr_matrix(nx.to_scipy_sparse_array(G))

P, a, b, g = get_separate_perm(A)
A = (A.toarray()[P, :])[:, P]

print("alpha,alpha = ", np.linalg.norm(A[:a, :a], ord='fro'))
print("beta,beta = ", np.linalg.norm(A[a:(a+b), a:(a+b)], ord='fro'))
print("gamma,gamma = ", np.linalg.norm(A[(a+b):, (a+b):], ord='fro'))
print("alpha,beta = ", np.linalg.norm(A[:a, a:(a+b)], ord='fro'))
print("beta,alpha = ", np.linalg.norm(A[a:(a+b), :a], ord='fro'))
