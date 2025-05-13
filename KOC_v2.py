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


if __name__ == "__main__":
    repeat_times = 1
    cmp_times = []
    cmp_costs = []
    cmp_nmi_level2 = []
    cmp_nmi_level3 = []
    num_nodes_range = list(range(100, 600, 500))
    for num_nodes in num_nodes_range:
        k = 4
        num_in = int(0.9 * num_nodes)
        data = list(range(1, num_in + 1))
        sizes = []
        remaining = num_in
        for _ in range(k - 1):
            size = random.randint(num_in // k - num_in // 100, num_in // k + num_in // 100)
            sizes.append(size)
            remaining -= size
        sizes.append(remaining)
        OC = []
        start_index = 0
        for size in sizes:
            OC.append(data[start_index:start_index + size])
            start_index += size
        dic = dict()
        tmp = 0
        for i in range(k):
            for j in range(i + 1, k):
                dic[tmp] = (i, j)
                tmp += 1
        for node in range(num_in + 1, num_nodes + 1):
            a, b = dic[node % tmp]
            OC[a].append(node)
            OC[b].append(node)
        for p_in, p_out1, p_out2 in [(3e-1, 5e-2, 1e-2)]:
            print("============")
            print("num_nodes:{}, p_in:{}, p_out1:{}, p_out2:{}".format(num_nodes, p_in, p_out1, p_out2))
            probs = {0: p_out2}
            for i in range(k):
                probs[1 << i] = p_in
            for i in range(k // 2):
                probs[((1 << 0) | (1 << 1)) << (i << 1)] = p_out1
            with open("data/visualize/gt_level2.txt", "w") as f:
                for i in range(k // 2):
                    c = set(OC[2 * i]) | set(OC[2 * i + 1])
                    f.write(' '.join([str(item) for item in c]))
                    f.write('\n')
                if k % 2:
                    f.write(' '.join([str(item) for item in OC[-1]]))
                    f.write('\n')

            for _ in range(repeat_times):
                # print(G)
                G = generate_overlapping_block_model(OC, probs, edges_path="data/visualize/edges.txt",
                                                     gt_path="data/visualize/gt_level3.txt")
                result, dag_edges = find_k_clusters(G, k, overlap=True, path="data/visualize/ex_level3.txt")
                dag = HOC()
                for u in dag_edges.keys():
                    for v in dag_edges[u]:
                        dag.add_edge(u, v)
                print(dag.HOC_cost([(u, v, 1) for (u, v) in G.edges()]))
                nmi_level2 = 0
                nmi_level3 = getnmi("data/visualize/ex_level3.txt", "data/visualize/gt_level3.txt")
                with open("data/visualize/ex_level2.txt", "w") as f:
                    S = dag.get_level(2)
                    for node in S:
                        c = dag.node2leaf[node]
                        f.write(' '.join([str(item) for item in c]))
                        f.write('\n')
                nmi_level2 = getnmi("data/visualize/ex_level2.txt", "data/visualize/gt_level2.txt")
                print(nmi_level2, nmi_level3)
