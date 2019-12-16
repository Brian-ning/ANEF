import numpy as np
import Conver2graph as c2g
import multiplexcd as mcd
import Community as cd
import Word2Vec
import random_walk_sampler as MHRW
import Participation_Coefficient as MPC
import ForestFireCross
from functools import partial
from multiprocessing import Pool as ThreadPool
import Node2Vec_LayerSelect
import copy
from ohmnet import ohmnet


def get_alias_edges(g, src, dest, p=1, q=1):
    probs = []
    for nei in sorted(g.neighbors(dest)):
        if nei == src:
            probs.append(1 / p)
        elif g.has_edge(nei, src):
            probs.append(1)
        else:
            probs.append(1 / q)
    norm_probs = [float(prob) / sum(probs) for prob in probs]
    return get_alias_nodes(norm_probs)


def get_alias_nodes(probs):
    l = len(probs)
    a, b = np.zeros(l), np.zeros(l, dtype=np.int)
    small, large = [], []

    for i, prob in enumerate(probs):
        a[i] = l * prob
        if a[i] < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        sma, lar = small.pop(), large.pop()
        b[sma] = lar
        a[lar] += a[sma] - 1.0
        if a[lar] < 1.0:
            small.append(lar)
        else:
            large.append(lar)
    return b, a


def preprocess_transition_probs(g, directed=False, p=1, q=1):
    alias_nodes, alias_edges = {}, {}
    for node in g.nodes():
        probs = [g[node][nei]["weight"] for nei in sorted(g.neighbors(node))]
        norm_const = sum(probs)
        norm_probs = [float(prob) / norm_const for prob in probs]
        alias_nodes[node] = get_alias_nodes(norm_probs)

    if directed:
        for edge in g.edges():
            alias_edges[edge] = get_alias_edges(g, edge[0], edge[1], p, q)
            # print(alias_edges[edge])
    else:
        for edge in g.edges():
            alias_edges[edge] = get_alias_edges(g, edge[0], edge[1], p, q)
            alias_edges[(edge[1], edge[0])] = get_alias_edges(g, edge[1], edge[0], p, q)

    return alias_nodes, alias_edges


def node2vec_walk(g, alias_nodes, alias_edges, walk_length, start):
    path = [start]
    walk_length = walk_length
    while len(path) < walk_length:
        node = path[-1]
        neis = sorted(g.neighbors(node))
        if len(neis) > 0:
            if len(path) == 1:
                l = len(alias_nodes[node][0])
                idx = int(np.floor(np.random.rand() * l))
                if np.random.rand() < alias_nodes[node][1][idx]:
                    path.append(neis[idx])
                else:
                    path.append(neis[alias_nodes[node][0][idx]])
            else:
                prev = path[-2]
                l = len(alias_edges[(prev, node)][0])
                idx = int(np.floor(np.random.rand() * l))
                if np.random.rand() < alias_edges[(prev, node)][1][idx]:
                    path.append(neis[idx])
                else:
                    path.append(neis[alias_edges[(prev, node)][0][idx]])
        else:
            break
    return path

# Caulate modularity
def CaulateModulary(nx_graphs, node_num, matrixs, mappings):
    iggraph = []
    type_list = ["s" for i in range(len(nx_graphs))]
    for net in nx_graphs:
        temp = c2g.nx_2_igraph(net)
        iggraph.append(temp)
    distance = np.zeros((node_num, node_num))
    for keyi, valuei in mappings.items():
        for keyj, valuej in mappings.items():
            if keyi != keyj:
                if keyi.isdecimal() and keyj.isdecimal():
                        distance[int(keyi)][int(keyj)] = np.dot(matrixs[valuei], matrixs[valuej])
    # paramet = {'linkage':'single', 'clusterNum':2}
    node_l = list(cd.cluster_distance_matrix(distance, "spectral"))
    node_label = []
    for gs in nx_graphs:
        nodelist = [int(vs) for vs in list(gs.nodes())]
        nodelist.sort()
        node_label.extend([node_l[vsid-1] for vsid in nodelist])

    B, om = mcd.get_modularity_matrix(iggraph, 1.0, type_list)  # 生成矩阵
    Q = mcd.multiplex_modularity(B, om, node_label)
    return Q


def CaulateMapping(nx_graphs, m_graph, p, q, num_walks, walk_length, r, ff):
    merge_networks = m_graph
    merge_nodes = merge_networks.nodes()
    allnodes = m_graph.nodes()

    # # # 对比试验 1： Deepwalk of merge network embedding
    # deep_walks = []
    # alias_nodes, alias_edges = preprocess_transition_probs(merge_networks)
    # parall_deepwalk = partial(node2vec_walk, merge_networks, alias_nodes, alias_edges, walk_length)
    # with ThreadPool(processes=3) as pool:
    #     for n_w in range(len(nx_graphs)):
    #         deep_walks.extend(pool.map(parall_deepwalk, merge_nodes))
    # deep_walks = [str(v) for u in deep_walks for v in u]
    #
    # # 对比试验 2： node2vec of merge network embedding
    # n2v_walks = []
    # alias_nodes, alias_edges = preprocess_transition_probs(merge_networks, False, p, q)
    # parall_Node2vec = partial(node2vec_walk, merge_networks, alias_nodes, alias_edges, walk_length)
    # with ThreadPool(processes=3) as pool:
    #     for n_w in range(len(nx_graphs)):
    #         n2v_walks.extend(pool.map(parall_Node2vec, merge_nodes))
    # n2v_walks = [str(v) for u in n2v_walks for v in u]
    #
    # # 对比试验 3：Node2vec Random
    # MK_G = Node2Vec_LayerSelect.Graph(nx_graphs, m_graph, p, q, r)
    # MK_G.preprocess_transition_probs(1)
    # MK_walks = MK_G.simulate_walks(len(nx_graphs), walk_length)
    # MK_words = []
    # for walk in MK_walks:
    #     MK_words.extend([str(step) for step in walk])
    #
    # # 对比试验 4: Ohmnet
    # ohmnet_walks = []
    # LG = copy.deepcopy(nx_graphs)
    # on = ohmnet.OhmNet(LG, p=p, q=q, num_walks=len(nx_graphs),
    #                    walk_length=walk_length, dimension=100,
    #                    window_size=10, n_workers=8, n_iter=5, out_dir='.')
    # count = 0
    # orignal_walks = []
    # for ns in on.embed_multilayer():
    #     count += 1
    #     orignal_walks.append(ns)
    #     on_walks = [n.split("_")[2] for n in ns]
    #     ohmnet_walks.extend([str(step) for step in on_walks])
    #     ohmnet_walks = ohmnet_walks[:len(allnodes)*walk_length*num_walks]
    # print("-----------------------DONE--------------------------------%i" % len(ohmnet_walks))

    # FFSN = []
    MHWL = []
    visited = []
    Alg_matrix = []
    Alg_mapping = []
    # Stem 5：多层网络的表示学习方法Node_layers
    # 将网络中的边权转化为1，然后进行网络的重构
    # 节点序列预处理

    nodeinfluence, node_exit = MPC.Multiplex_PartC(m_graph, nx_graphs)  # 重要性的定义
    ExpansionSample_MHRW = partial(MHRW.random_walk_sampler, nx_graphs, node_exit, walk_length, nodeinfluence, True)
    # ExpansionSample_FFS = partial(ForestFireCross.forest_fire_sampling, nx_graphs, node_exit, walk_length, nodeinfluence, ff)
    with ThreadPool(processes=4) as pool:
        for i in range(len(nx_graphs)):
            MHWL.extend(j for i in pool.map(ExpansionSample_MHRW, allnodes) for j in i)
            # FFSN.extend(j for i in pool.map(ExpansionSample_FFS, allnodes) for j in i)

    # 节点序列预处理
    # visited.append([str(ite) for ite in deep_walks])
    # visited.append([str(ite) for ite in n2v_walks])
    # visited.append([str(ite) for ite in MK_words])
    # visited.append([str(ite) for ite in ohmnet_walks])
    visited.append([str(ite) for ite in MHWL])
    # visited.append([str(ite) for ite in FFSN])
    alg_num = len(visited)
    # 对生成的序列进行词向量的学习
    for setps in range(alg_num):
        alg_learning = Word2Vec.Learn(visited[setps])
        try_matrix, try_mapping = alg_learning.train()
        Alg_matrix.append(try_matrix)
        Alg_mapping.append(try_mapping)
        print("-----------DONE-------------Algorithm [%i]:%i" % (setps, len(visited[setps])))
    return Alg_matrix, Alg_mapping
