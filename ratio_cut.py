import scipy as sp
import sklearn.cluster
import networkx as nx


def RatioCut(G: nx.Graph, k: int = 2):
    # print(G)
    id2node = dict()
    for id, nd in enumerate(G.nodes):
        id2node[id] = nd
    laplacian_matrix = nx.normalized_laplacian_matrix(G)
    # print(nx.laplacian_matrix(G))
    eigs, Uk = sp.sparse.linalg.eigsh(laplacian_matrix, k=k, which="SM", return_eigenvectors=True, tol=1E-2,
                                      maxiter=5000)
    kmeansFuc = sklearn.cluster.KMeans(n_clusters=k)
    kmeansFuc.fit(Uk)
    x = kmeansFuc.labels_
    Result = [[], []]
    for i in range(0, len(x)):
        Result[x[i]].append(id2node[i])
    return Result


if __name__ == "__main__":
    G = nx.karate_club_graph()
    print(G.edges(data=True))
    RatioCut(G)
