#!/usr/bin/python
# -*- coding: utf-8 -*-

import Reader
import CaulateQ as cq
import networkx as nx
import pandas as pd


class Mergeing_vec_N2V:
    def __init__(self, path, p, q, num_walks, walk_length, r, ff):
        self.path = path
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.r = r
        self.ff = ff

    def run(self):
        path = self.path

        # Step 1: reading and sampling graphs
        t_graph, tnx_graphs, total_edges = Reader.multi_readG_with_Merg(path)
        # H = nx.convert_node_labels_to_integers(t_graph)
        newlabels = range(nx.number_of_nodes(t_graph))
        nodelist = list(t_graph.nodes())
        relabels = zip(nodelist, newlabels)
        nvDict = dict((name, value) for name, value in relabels)
        m_graph = nx.relabel_nodes(t_graph, nvDict)
        nx_graphs = []
        for layer in tnx_graphs:
            nx_graphs.append(nx.relabel_nodes(layer, nvDict))

        # #   #
        # # # # 实验 2：MELL实现多层网络的节点表示学习，WWW’2018
        # edge_list = list(m_graph.edges())
        #
        # restru_test_edges = []
        # restru_test_edges.append([[nodes.index(e[0]), nodes.index(e[1])] for e in edge_list])
        #
        # L = len(nx_graphs)
        # N = max([int(n) for n in m_graph.nodes()])+1
        # N = max(N, m_graph.number_of_nodes()) # 为了构造邻接矩阵需要找到行的标准
        # directed = True
        # d = 100
        # k = 3
        # lamm = 10
        # beta = 1
        # gamma = 1
        # MELL_wvecs = MELL_model(L, N, directed, edge_list, d, k, lamm, beta, gamma)
        # MELL_wvecs.train(200) # 之前是500，但是有的数据集500会报错，因此设置为30
        # NH_wvecs = {}
        # NT_wvecs = {}
        # for vh in nodes:
        #     vec = np.asarray([0]*d)
        #     for l in range(len(nx_graphs_sampled)):
        #         vec = vec + MELL_wvecs.resVH[l, vh]
        #     NH_wvecs[vh] = vec
        # for vt in nodes:
        #     vec = np.asarray([0]*d)
        #     for l in range(len(nx_graphs_sampled)):
        #         vec = vec+ MELL_wvecs.resVT[l, vt]
        #     NT_wvecs[vt] = vec

        print("-------------------START Modularity----------------")
        Q = []
        matrixs, mappings = cq.CaulateMapping(nx_graphs, m_graph, self.p, self.q, self.num_walks, self.walk_length, self.r, self.ff)
        alg_num = len(matrixs)
        node_num = len(m_graph.nodes())
        for alg in range(alg_num):
            Q.append(cq.CaulateModulary(nx_graphs, node_num, matrixs[alg], mappings[alg]))
        print("The Modularity of Algorithm:")
        print(Q)
        df = pd.DataFrame(Q)
        df.to_csv('end_result.csv', float_format='%.4f', mode='a')
        print("-------------------END Modularity--------------------")
