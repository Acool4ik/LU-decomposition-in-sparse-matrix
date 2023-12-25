import numpy as np
import networkx as nx


def kernighan_lin(sparse_matrix):
    G = nx.Graph(sparse_matrix)
    mat = nx.adjacency_matrix(G, weight=None)
    G = nx.Graph(mat)

    partition_a, partition_b = nx.community.kernighan_lin_bisection(G, max_iter=1000, seed=911)

    separator = set()
    connecting_edges = []

    for edge in G.edges():
        if (edge[0] in partition_a and edge[1] in partition_b) or (edge[0] in partition_b and edge[1] in partition_a):
            connecting_edges.append(edge)

    for edge in connecting_edges:
        if (edge[0] not in separator) and (edge[1] not in separator):
            separator.add(edge[0])
            separator.add(edge[1])

    partition_a -= separator
    partition_b -= separator

    partition_a = np.array(list(partition_a))
    partition_b = np.array(list(partition_b))
    separator = np.array(list(separator))

    a, b, g = len(partition_a), len(partition_b), len(separator)
    start = 0
    P = np.zeros(G.number_of_nodes(), dtype=np.int32)
    if a > 0: P[partition_a] = np.arange(start, start + a)
    start += a
    if b > 0: P[partition_b] = np.arange(start, start + b)
    start += b
    if g > 0: P[separator] = np.arange(start, start + g)
    P = P.argsort()
    return P, a, b, g
