#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, pickle, sys
import networkx as nx
import scipy.io as scio

def single_readG(path):
    if os.path.isfile(path) and path.endswith(".pickle"):
        g_need = pickle.load(open(path, "rb"))
        #g_need = max(nx.connected_component_subgraphs(g), key=len)
        return g_need
    else:
        sys.exit("##cannot find the pickle file from give path: " + path + "##")


def multi_readG_with_Merg(path):
    if os.path.isdir(path):  # Judge whether this path is folder
        files = os.listdir(path)  # Get the file name list under this folder
        nx_graphs = []  # inistall the variable
        m_graph = -1
        total_edges = 0  # The total number of edges
        for name in files:
            if name.endswith("pickle"):  # Checking the file name
                if "merged_graph" in name:
                    m_graph = single_readG(path + '/' + name)
                else:
                    g_need = pickle.load(open(path + '/' + name, "rb"))
                    nx_graphs.append(g_need)
                    total_edges += len(g_need.edges())
        return m_graph, nx_graphs, total_edges
