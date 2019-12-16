# -*- coding: utf-8 -*-
import random as rd
import networkx as nx


def prim(graph, root, walk_length):
    nodels = list(graph.nodes())
    nodels.remove(root)
    nodeseq = [root]
    visited = [root]
    path = []
    nextn = None

    while nodels:
        if len(nodeseq) == walk_length:
            break
        distance = -1
        for s in visited:
            for d in graph.neighbors(s):
                if d in visited or s == d:
                    continue
                if float(graph.edges[s, d]["weight"]) > distance:
                    distance = graph.edges[s, d]["weight"]
                    pre = s
                    nextn = d
        path.append((pre, nextn))
        if pre == nodeseq[-1]:
            nodeseq.append(nextn)
        else:
            count = -1
            tunseq = []
            while pre != nodeseq[count-1]:
                tunseq.append(nodeseq[count-1])
                count = count - 1
            if len(tunseq) != 0:
                nodeseq.extend([item for item in tunseq])
            nodeseq.append(pre)
            nodeseq.append(nextn)
        visited.append(nextn)
        nodels.remove(nextn)
    return nodeseq

