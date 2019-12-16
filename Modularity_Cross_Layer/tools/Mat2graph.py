#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys, pickle
import scipy.io as scio
import networkx as nx

def loadMat(path):
    if os.path.isdir(path):
        print("reading from " + path + "......")
        files = os.listdir(path)
        counter = 1
        for name in files:
            if name.endswith(".mat"):
                print("found file " + name + "...")
                data = scio.loadmat(path+name)
                matrix = data['A'+str(counter)]
                Mat2edge(matrix, path+name)
                counter += 1

        print("ALL DONE")

    else:
        sys.exit("##input path is not a directory##")

def Mat2edge(matrix, name):
    #graph = nx.Graph()
    l = 0
    f = open(name+'_edgelist.txt', 'w+')
    for i in range(0, len(matrix)):
        for j in range(l+1, len(matrix)):
            if matrix[i][j] > 0:
                #graph.add_edge(i, j)
                f.write(str(i) + ' ' + str(j) + '\n')
        l += 1


    #pickle.dump(graph, open(name+'.nx_graph.pickle', '+wb'))


def read_f(filename):
    if os.path.isfile(filename) and filename.endswith(".edges"):
        print("reading from " + filename + "......")
        graph_dict = {}
        for line in open(filename):
            line_sp = line.split()
            if len(line_sp) < 4:
                (layer_id, src, dst) = line_sp
                if layer_id not in graph_dict.keys():
                    graph_dict[layer_id] = nx.DiGraph(name=layer_id)
                    graph_dict[layer_id].add_edge(src, dst , weight = 1)
                else:
                    graph_dict[layer_id].add_edge(src, dst , weight = 1)
            else:
                (layer_id, src, dst, weight,) = line_sp
                if layer_id not in graph_dict.keys():
                    graph_dict[layer_id] = nx.DiGraph(name=layer_id)
                    graph_dict[layer_id].add_edge(src, dst, weight = float(weight))
                else:
                    graph_dict[layer_id].add_edge(src, dst, weight = float(weight))

        List_graph = list(graph_dict.values())
        for item in List_graph:
            pickle.dump(item, open(item.name + 'nx_graph.pickle', '+wb'))
        # for i in graph_dict:
        #     f = open(i+'.txt', 'w+')
        #     for edge in graph_dict[i].edges():
        #         f.write(edge[0] + ' ' + edge[1] + '\n')


if __name__ == '__main__':
    #loadMat(sys.argv[1])
    read_f("Celegans.edges")
