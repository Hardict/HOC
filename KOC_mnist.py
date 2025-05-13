import fast_algorithm_for_2OC
import networkx as nx
import datetime
import random
import time
import numpy as np
import subprocess
import re
import os
import matplotlib.pyplot as plt
from HOC import HOC


def read_network(path):
    f = open(path, "r").readlines()
    edges = set()
    nodes = set()
    for edge in f:
        a, b = edge.split()
        edges.add((int(a), int(b)))
        nodes.add(int(a))
        nodes.add(int(b))
    return list(nodes), list(edges)


def calc_delta(G: nx.Graph, cluster):
    new_G: nx.Graph = G
    new_C1, new_C2 = fast_algorithm_for_2OC.greedy_select_overlap_greedy(G)
    S1 = set(new_C1)
    S2 = set(new_C2)
    B = S1.intersection(S2)
    A = S1 - S2
    C = S2 - S1

    H1 = nx.volume(new_G, cluster) / 2 * len(new_G)
    wA = nx.volume(new_G, A) / 2
    wB = nx.volume(new_G, B) / 2
    wC = nx.volume(new_G, C) / 2
    wAB = 0
    wBC = 0
    wAC = 0
    for u, v in G.edges(data=False):
        w = 1
        if 'weight' in G[u][v]:
            w = G[u][v]['weight']
        if (u in A and v in B) or (u in B and v in A):
            wAB += w
        if (u in A and v in C) or (u in C and v in A):
            wAC += w
        if (u in B and v in C) or (u in C and v in B):
            wBC += w

    H2 = (wA + wAB) * (len(A) + len(B)) + (wC + wBC) * (len(B) + len(C)) + wAC * len(cluster) + 1 / 2 * wB * (
            len(cluster) + len(B))
    delta = (H1 - H2) / H1
    return delta


def new_id(start_val: int = 1):
    """
    利用yield分配Node的ID
    :return:
    """
    i = start_val
    while True:
        yield i
        i += 1


id_generator = 1


def find_k_clusters(G: nx.Graph, k, batch_size=1 << 5, max_move_times=5, path="data/ex.txt", overlap=True):
    dag_edges = dict()
    id_generator = new_id()

    name_root = "internal_" + str(next(id_generator))
    if k > 1:
        dag_edges[name_root] = set()
        C1, C2 = fast_algorithm_for_2OC.greedy_select_overlap_greedy(G, batch_size, max_move_times, overlap)
        name_C1 = "internal_" + str(next(id_generator))
        dag_edges[name_C1] = set(C1)
        dag_edges[name_root].add(name_C1)
        name_C2 = "internal_" + str(next(id_generator))
        dag_edges[name_C2] = set(C2)
        dag_edges[name_root].add(name_C2)
        result = [(C1, name_C1), (C2, name_C2)]
        cnt_cluster = 2

        dic = {}
        for i, (cluster, name_cluster) in enumerate(result):
            new_G: nx.Graph = G.subgraph(cluster)
            delta = calc_delta(new_G, cluster)
            dic[tuple(cluster)] = (delta, name_cluster)
        # print("step2")
        while cnt_cluster < k:
            delta_max = -1e9
            name_max = ""
            id_max = 0
            for i, (cluster, name_cluster) in enumerate(result):
                (delta, name_cluster) = dic[tuple(cluster)]
                if delta_max < delta:
                    delta_max = delta
                    name_max = name_cluster
                    id_max = i

            new_G: nx.Graph = G.subgraph(result[id_max][0])
            del result[id_max]
            # for node in new_G.nodes():
            #     if node in dag_edges[name_max]:
            #         dag_edges[name_max].remove(node)
            dag_edges[name_max] = set()

            new_C1, new_C2 = fast_algorithm_for_2OC.greedy_select_overlap_greedy(new_G, batch_size, max_move_times,
                                                                                 overlap)
            name_C1 = "internal_" + str(next(id_generator))
            dag_edges[name_C1] = set(new_C1)
            dag_edges[name_max].add(name_C1)
            name_C2 = "internal_" + str(next(id_generator))
            dag_edges[name_C2] = set(new_C2)
            dag_edges[name_max].add(name_C2)
            result.append((new_C1, name_C1))
            result.append((new_C2, name_C2))
            dic[tuple(new_C1)] = (calc_delta(G.subgraph(new_C1), new_C1), name_C1)
            dic[tuple(new_C2)] = (calc_delta(G.subgraph(new_C2), new_C2), name_C2)
            cnt_cluster += 1
        # for item in result:
        #     print(len(item), item)
    else:
        result = [(G.nodes(), name_root)]
        dag_edges[name_root] = set(G.nodes())
    # print(dag_edges)
    f = open(path, "w")
    for c, name in result:
        for node in c:
            f.write(f"{node} ")
        f.write('\n')
    f.close()
    return result, dag_edges


def generate_overlapping_block_model(OC, probs, edges_path="data/edges.txt", gt_path="data/gt.txt", sparse=False):
    # OC: list: 每个cluster点
    # probs: dict(bitset->prob): bitset表示其overlap设计点集，空集表示外连边

    G = nx.Graph()
    for C in OC:
        for node in C:
            G.add_node(node)

    if 0 in probs.keys():
        nodes = list(G.nodes())
        if sparse:
            num_nodes = len(nodes)
            num_edges = int(probs[0] * num_nodes ** 2 / 2)
            for _ in range(num_edges):
                u = nodes[np.random.randint(low=num_nodes)]
                v = nodes[np.random.randint(low=num_nodes)]
                if u != v:
                    G.add_edge(u, v)
        else:
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if i > j:
                        if np.random.rand() < probs[0]:
                            G.add_edge(u, v)
                    else:
                        break
    for state in probs.keys():
        if state == 0:
            continue
        nodes = set()
        tmp = state
        for i in range(len(OC)):
            if tmp & 1:
                nodes |= set(OC[i])
            # else:
            #     nodes = nodes - set(OC[i])
            tmp >>= 1
        # print(nodes, probs[state])
        nodes = list(nodes)
        if sparse:
            num_nodes = len(nodes)
            num_edges = int(probs[state] * num_nodes ** 2 / 2)
            for _ in range(num_edges):
                u = nodes[np.random.randint(low=num_nodes)]
                v = nodes[np.random.randint(low=num_nodes)]
                if u != v:
                    G.add_edge(u, v)
        else:
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if i > j and np.random.rand() < probs[state]:
                        G.add_edge(u, v)
    f = open(edges_path, 'w')
    for u, v in G.edges(data=False):
        f.write(f"{u} {v}\n")
    f.close()
    f = open(gt_path, "w")
    for c in OC:
        f.write(' '.join([str(item) for item in c]))
        f.write('\n')
    f.close()
    # print(OC)
    return G


def getnmi(ex_path, gt_path):
    ret = 0
    nmi = subprocess.check_output(
        './Overlapping-NMI-master/onmi.exe {} {}'.format(ex_path, gt_path),
        shell=False)
    # matches = re.findall(r'\d*\.\d+|\d+', nmi.decode('utf-8'))
    match = re.search(r"NMI<Sum>:\s+([\d.eE+-]+)", nmi.decode("utf-8"))
    if match:
        result = float(match.group(1))  # 转换为浮点数
        ret = result
    return ret


def create_graph_from_edge_weights(edge_weights, k):
    """
    根据边权矩阵生成图，每个节点只保留最近的 k 个邻居
    :param edge_weights: 边权矩阵
    :param k: 每个节点保留的最近邻居数量
    :return: 生成的图
    """
    num_nodes = edge_weights.shape[0]
    G = nx.Graph()

    # 根据边权矩阵找到每个节点的最近的 k 个邻居
    for i in range(num_nodes):
        # 找到第 i 个节点的最近的 k 个邻居索引
        edges = [(i, j, edge_weights[i, j]) for j in range(num_nodes) if j != i]
        # 根据边权进行排序
        sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
        # 保留与节点i相连的最大k条边
        selected_edges = sorted_edges[:k]
        # 将保留的边添加到图中
        G.add_weighted_edges_from(selected_edges)

    return G


from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def save_data(file_path, data):
    """
    保存数据到txt文件
    :param file_path: 文件路径
    :param data: 数据列表
    """
    with open(file_path, 'w') as file:
        for item in data:
            lis = item.tolist()
            if type(lis) == float:
                lis = [lis]
            file.write(' '.join(map(str, lis)))
            file.write('\n')


def save_graph(file_path, G: nx.Graph):
    with open(file_path, 'w') as file:
        for (u, v) in G.edges(data=False):
            w = 1
            if 'weight' in G[u][v]:
                w = G[u][v]['weight']
            file.write('{} {} {}\n'.format(u, v, G[u][v]['weight']))


if __name__ == "__main__":
    st_time = time.strftime('%Y-%m-%d %H-%M-%S %A')
    num_clusters = 2
    log_path = "log.txt".format(num_clusters, time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()))
    fp = open(log_path, mode="w", encoding="utf-8")

    suffix = '17'
    edges_path = "data/edges_mnist.txt"
    gt_path = "data/gt_mnist_{}.txt".format(suffix)
    ex_path = "data/ex_mnist_{}.txt".format(suffix)

    # 读取txt文件
    data = np.loadtxt('mnist_features_total.txt')
    label = np.loadtxt('mnist_labels_total.txt')

    target_labels = [1, 7]
    filtered_data = []
    filtered_label = []
    for vec, lb in zip(data, label):
        if int(lb) in target_labels:
            filtered_data.append(vec)
            filtered_label.append(lb)
    save_data('mnist_features_total_{}.txt'.format(suffix), filtered_data)
    save_data('mnist_labels_total_{}.txt'.format(suffix), filtered_label)

    with open(gt_path, "w") as fp:
        dic = dict()
        for i, label in enumerate(filtered_label):
            if label not in dic:
                dic[label] = set()
            dic[label].add(i)
        for label in dic:
            fp.write(' '.join([str(item) for item in dic[label]]))
            fp.write('\n')

    # pca = PCA(n_components=20)  # 降到 50 维
    # pca_data = pca.fit_transform(filtered_data)
    # # 计算两两之间的距离
    # distances = pdist(pca_data)
    # # 将距离转换为对称矩阵
    # distance_matrix = squareform(distances)
    # # 打印距离矩阵
    # print(distance_matrix)
    # sigma = 2500.0
    # # sigma = 10
    # edge_weights = np.exp(-distance_matrix ** 2 / (2 * sigma ** 2))
    # print("边权矩阵：")
    # print(edge_weights)
    #
    # G = create_graph_from_edge_weights(edge_weights, 100)
    # save_graph('data/mnist_total_{}_edges.txt'.format(suffix), G)
    # # print(G.edges(data=True))
    #
    # # nodes, edges = read_network(edges_path)
    # # G = nx.Graph()
    # # np.random.shuffle(edges)
    # # G.add_weighted_edges_from([(str(u), str(v), 1) for u, v in edges])
    # # G.add_nodes_from(nodes)
    #
    # find_k_clusters(G, num_clusters, path=ex_path)
    # end_time = time.strftime('%Y-%m-%d %H-%M-%S %A')
    # # print(len(sub_nodes), len(sub_edges))
    # print("node size:{}, number of edges:{}".format(len(G.nodes), len(G.edges)), file=fp)
    # print("start time:{}, end time:{}".format(st_time, end_time), file=fp)
    # time_difference = datetime.datetime.strptime(end_time, '%Y-%m-%d %H-%M-%S %A') - \
    #                   datetime.datetime.strptime(st_time, '%Y-%m-%d %H-%M-%S %A')
    # seconds_difference = time_difference.total_seconds()
    # print("seconds:{}".format(seconds_difference), file=fp)
    print(getnmi(ex_path, gt_path))
