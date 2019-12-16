import networkx as nx
import random


def Multiplex_PartC(m_graph, nx_graphs):
    nodelist = list(m_graph.nodes()) # 获得节点列表
    nodelist.sort()

    node_exit = {} # 初始化节点激活所在的层
    layer_num = len(nx_graphs) # 层的个数
    part_coe = {} # 初始化系数字典

    for v in nodelist:
        num_vertex = []
        v_degree_layer = []
        node_layer = []
        for l1 in range(layer_num):
            if v in nx_graphs[l1].nodes():
                v_degree_layer.append(nx_graphs[l1].degree(v))
                node_layer.append(l1)
                for l2 in range(layer_num):
                    if v in nx_graphs[l2].nodes():
                        num_vertex.append(len(set(nx_graphs[l1].neighbors(v)).intersection(nx_graphs[l2].neighbors(v))))
        part_coe[v] = sum(num_vertex)/ (layer_num * sum(v_degree_layer))
        node_exit[v] = node_layer

    return part_coe, node_exit
