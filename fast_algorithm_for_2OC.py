import random
import bisect
from ratio_cut import *
import networkx as nx


def cal_cost(A, B, C, G: nx.Graph):
    node_to_com = {}
    w_A, w_B, w_C, w_A_B, w_B_C = 0, 0, 0, 0, 0
    for x in A:
        node_to_com[x] = "A"
    for x in B:
        node_to_com[x] = "B"
    for x in C:
        node_to_com[x] = "C"
    for u, v in G.edges(data=False):
        w = 1
        if 'weight' in G[u][v]:
            w = G[u][v]['weight']
        if node_to_com[u] == "A":
            if node_to_com[v] == "A":
                w_A += w
            elif node_to_com[v] == "B":
                w_A_B += w
        elif node_to_com[u] == "B":
            if node_to_com[v] == "A":
                w_A_B += w
            elif node_to_com[v] == "B":
                w_B += w
            else:
                w_B_C += w
        else:
            if node_to_com[v] == "B":
                w_B_C += w
            elif node_to_com[v] == "C":
                w_C += w
    cost = (w_A + w_A_B) * len(C) + (w_C + w_B_C) * len(A) + 0.5 * w_B * (len(A) + len(C))
    return cost


def create_node_to_setWeight(A, B, C, G: nx.Graph):
    node_to_setWeight = {}
    node_to_setName = {}
    near_label = {}
    for node in G.nodes:
        node_to_setWeight[node] = {}
        node_to_setWeight[node]['A'] = 0
        node_to_setWeight[node]['B'] = 0
        node_to_setWeight[node]['C'] = 0
        near_label[node] = set()
    for node in A:
        node_to_setName[node] = 'A'
    for node in B:
        node_to_setName[node] = 'B'
    for node in C:
        node_to_setName[node] = 'C'
    for u, v in G.edges(data=False):
        near_label[u].add(v)
        near_label[v].add(u)
        w = 1
        if 'weight' in G[u][v]:
            w = G[u][v]['weight']
        if u in A:
            node_to_setWeight[v]['A'] += w
        elif u in B:
            node_to_setWeight[v]['B'] += w
        else:
            node_to_setWeight[v]['C'] += w
        if v in A:
            node_to_setWeight[u]['A'] += w
        elif v in B:
            node_to_setWeight[u]['B'] += w
        else:
            node_to_setWeight[u]['C'] += w
    return node_to_setWeight, node_to_setName, near_label


def init_nodes(G: nx.Graph):
    X, Y = RatioCut(G)
    A = set(X)
    B = set()
    C = set(Y)
    return A, B, C


if_time = 1


def cal_move_node_delta(action, node, node_to_setWeight, setWeight, A, B, C, G: nx.Graph):
    test = 0
    delta = 0
    w_node_A = node_to_setWeight[node]['A']
    w_node_B = node_to_setWeight[node]['B']
    w_node_C = node_to_setWeight[node]['C']
    w_A_plus_B = setWeight['A']['A'] + setWeight['A']['B'] + setWeight['B']['B']
    w_B_plus_C = setWeight['B']['B'] + setWeight['B']['C'] + setWeight['C']['C']
    w_B = setWeight['B']['B']
    if test:
        origin_cost = cal_cost(A, B, C, G)
    if action == "A-B":
        delta = -0.5 * w_node_B * len(C) + (0.5 * w_node_B + w_node_C) * len(A) - (
                w_B_plus_C + 0.5 * w_node_B + w_node_C - 0.5 * w_B)
        if test:
            if delta != cal_cost(A - {node}, B | {node}, C, G) - origin_cost:
                print("no")
    elif action == "A-C":
        delta = (-w_node_A - w_node_B) * len(C) + (w_node_B + w_node_C) * len(A) + (
                w_A_plus_B - w_B_plus_C - w_node_A - 2 * w_node_B - w_node_C)
        if test:
            if delta != cal_cost(A - {node}, B, C | {node}, G) - origin_cost:
                print("no")
    elif action == "B-A":
        delta = 0.5 * w_node_B * len(C) + (-0.5 * w_node_B - w_node_C) * len(A) + (
                w_B_plus_C - 0.5 * w_node_B - w_node_C - 0.5 * w_B)
        if test:
            if delta != cal_cost(A | {node}, B - {node}, C, G) - origin_cost:
                print("no")
    elif action == "B-C":
        delta = 0.5 * w_node_B * len(A) + (-0.5 * w_node_B - w_node_A) * len(C) + (
                w_A_plus_B - 0.5 * w_node_B - w_node_A - 0.5 * w_B)
        if test:
            if delta != cal_cost(A, B - {node}, C | {node}, G) - origin_cost:
                print("no")
    elif action == "C-A":
        delta = (-w_node_C - w_node_B) * len(A) + (w_node_B + w_node_A) * len(C) + (
                w_B_plus_C - w_A_plus_B - w_node_C - 2 * w_node_B - w_node_A)
        if test:
            if delta != cal_cost(A | {node}, B, C - {node}, G) - origin_cost:
                print("no")
    elif action == "C-B":
        delta = -0.5 * w_node_B * len(A) + (0.5 * w_node_B + w_node_A) * len(C) - (
                w_A_plus_B + 0.5 * w_node_B + w_node_A - 0.5 * w_B)
        if test:
            if delta != cal_cost(A, B | {node}, C - {node}, G) - origin_cost:
                print("no")
    return delta


def cal_setWeight(node_to_setName, G: nx.Graph):
    setWeight = {'A': {'A': 0, 'B': 0, 'C': 0}, 'B': {'B': 0, 'C': 0}, 'C': {'C': 0}}
    for u, v in G.edges(data=False):
        w = 1
        if 'weight' in G[u][v]:
            w = G[u][v]['weight']
        name_x, name_y = min(node_to_setName[u], node_to_setName[v]), max(node_to_setName[u], node_to_setName[v])
        setWeight[name_x][name_y] += w
    return setWeight


def update_move_node(A, B, C, node_to_setWeight, setWeight, node, action, near_label, node_to_setName):
    if action == "A-B":
        A.remove(node)
        B.add(node)
        node_to_setName[node] = 'B'
    elif action == 'A-C':
        A.remove(node)
        C.add(node)
        node_to_setName[node] = 'C'
    elif action == 'B-A':
        B.remove(node)
        A.add(node)
        node_to_setName[node] = 'A'
    elif action == 'B-C':
        B.remove(node)
        C.add(node)
        node_to_setName[node] = 'C'
    elif action == 'C-A':
        C.remove(node)
        A.add(node)
        node_to_setName[node] = 'A'
    elif action == 'C-B':
        C.remove(node)
        B.add(node)
        node_to_setName[node] = 'B'
    be, af, el = action[0], action[2], {'A', 'B', 'C'} - {action[0], action[2]}
    el = el.pop()
    for near_node in near_label[node]:
        node_to_setWeight[near_node][be] -= 1
        node_to_setWeight[near_node][af] += 1
    setWeight[af][af] += node_to_setWeight[node][af]
    setWeight[min(be, af)][max(be, af)] -= node_to_setWeight[node][af]
    setWeight[be][be] -= node_to_setWeight[node][be]
    setWeight[min(be, af)][max(be, af)] += node_to_setWeight[node][be]
    setWeight[min(be, el)][max(be, el)] -= node_to_setWeight[node][el]
    setWeight[min(af, el)][max(af, el)] += node_to_setWeight[node][el]


def move_nodes(A, B, C, G: nx.Graph, node_to_setWeight, setWeight, near_label, node_to_setName,
               batch_size=1 << 5,
               max_move_times=5,
               overlap=True):
    first, flag = 1, 0
    already = dict()
    batch_size = max(1, int(batch_size))
    # print(batch_size)
    while flag or first:
        to_move = []
        to_move_len = 0
        first = 0
        flag = 0
        for a in A:
            delta1 = cal_move_node_delta("A-B", a, node_to_setWeight, setWeight, A, B, C, G)
            delta2 = cal_move_node_delta("A-C", a, node_to_setWeight, setWeight, A, B, C, G)
            max_delta = max(delta1, delta2)
            max_action = 0
            if max_delta == delta1 and max_delta > 0 and overlap:
                max_action = "A-B"
            elif max_delta == delta2 and max_delta > 0:
                max_action = "A-C"
            if max_action != 0 and (a not in already or already[a] < max_move_times):
                if to_move_len < batch_size:
                    bisect.insort_left(to_move, (-max_delta, a, max_action))
                    to_move_len += 1
                    # print(to_move_len, k)
                elif to_move[-1][0] < max_delta:
                    bisect.insort_left(to_move, (-max_delta, a, max_action))
                    del to_move[-1]
        if overlap:
            for b in B:
                delta1 = cal_move_node_delta("B-A", b, node_to_setWeight, setWeight, A, B, C, G)
                delta2 = cal_move_node_delta("B-C", b, node_to_setWeight, setWeight, A, B, C, G)
                max_delta = max(delta1, delta2)
                max_action = 0
                if max_delta == delta1 and max_delta > 0:
                    max_action = "B-A"
                elif max_delta == delta2 and max_delta > 0:
                    max_action = "B-C"
                if max_action != 0 and (b not in already or already[b] < max_move_times):
                    if to_move_len < batch_size:
                        bisect.insort_left(to_move, (-max_delta, b, max_action))
                        to_move_len += 1
                    elif to_move[-1][0] < max_delta:
                        bisect.insort_left(to_move, (-max_delta, b, max_action))
                        del to_move[-1]
        for c in C:
            delta1 = cal_move_node_delta("C-A", c, node_to_setWeight, setWeight, A, B, C, G)
            delta2 = cal_move_node_delta("C-B", c, node_to_setWeight, setWeight, A, B, C, G)
            max_delta = max(delta1, delta2)
            max_action = 0
            if max_delta == delta1 and max_delta > 0:
                max_action = "C-A"
            elif max_delta == delta2 and max_delta > 0 and overlap:
                max_action = "C-B"
            if max_action != 0 and (c not in already or already[c] < max_move_times):
                if to_move_len < batch_size:
                    bisect.insort_left(to_move, (-max_delta, c, max_action))
                    to_move_len += 1
                elif to_move[-1][0] < max_delta:
                    bisect.insort_left(to_move, (-max_delta, c, max_action))
                    del to_move[-1]
        if len(to_move) != 0:
            if len(to_move) > batch_size:
                to_move = to_move[:batch_size]
            flag = 1
            for _, max_node, max_action in to_move:
                if max_node not in already:
                    already[max_node] = 0
                already[max_node] += 1
                update_move_node(A, B, C, node_to_setWeight, setWeight, max_node, max_action, near_label,
                                 node_to_setName)
    return [A, B, C]


# 优化版本
def greedy_select_overlap_greedy(G: nx.Graph, batch_size=1 << 5, max_move_times=5, overlap=True):
    global if_time
    import time
    # if if_time:
    #     print("stage0")
    #     t = time.time()
    A, B, C = init_nodes(G)
    setName_to_set = {}
    setName_to_set['A'] = A
    setName_to_set['B'] = B
    setName_to_set['C'] = C
    node_to_setWeight, node_to_setName, near_label = create_node_to_setWeight(A, B, C,
                                                                              G)  # node to weight, node to name
    setWeight = cal_setWeight(node_to_setName, G)
    # if if_time:
    #     print(f'coast:{time.time() - t:.4f}s')
    #     print("stage1")
    #     t = time.time()
    max_nodes = move_nodes(A, B, C, G, node_to_setWeight, setWeight, near_label, node_to_setName,
                           batch_size, max_move_times, overlap)
    A, B, C = set(max_nodes[0]), set(max_nodes[1]), set(max_nodes[2])
    # if if_time:
    #     print(f'coast:{time.time() - t:.4f}s')
    return sorted(A | B), sorted(B | C)
