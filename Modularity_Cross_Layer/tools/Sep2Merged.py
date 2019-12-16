#!/usr/bin/python
# -*- coding: utf-8 -*-

import networkx as nx
import Reader, sys, pickle

def merge_g(path):
    nx_graphs, _ = Reader.multi_readG(path)
    m_g = nx.Graph()
    for g in nx_graphs:
        for e in g.edges.data():
            if m_g.has_edge(e[0], e[1]):
                m_g[e[0]][e[1]]['weight'] = float(m_g[e[0]][e[1]]['weight']) + float(e[2]['weight'])
            else:
                m_g.add_edge(e[0], e[1], weight = e[2]['weight'])

    pickle.dump(m_g, open(path+'/'+'merged_graph.txtnx_graph.pickle', '+wb'))

def verify(path):
    m_graph, nx_graphs, _ = Reader.multi_readG_with_Merg(path)

    for g in nx_graphs:
        for edge in g.edges():
            if not m_graph.has_edge(*edge):
                print("INVALED MERGE", edge)
                print(m_graph.neighbors(edge[0]))
                print(m_graph.neighbors(edge[1]))
                print(m_graph.edges())
                print(g.edges())
                sys.exit("ERROR FOUND")

if __name__ == "__main__":
    merge_g('./')
    verify('./')
