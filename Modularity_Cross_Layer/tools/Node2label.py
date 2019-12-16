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
    if os.path.isfile(filename) and filename.endswith(".nodes"):
        print("reading from " + filename + "......")
        graph_dict = {}
        for line in open(filename):
            node_id = line.split(' ')[0]
            label = line.split(' ')[10]
            with open('node_label.txt', 'a+') as f:
                f.write(node_id + ' ' + label + '\n')

if __name__ == '__main__':
    #loadMat(sys.argv[1])
    read_f("CKM.nodes")
