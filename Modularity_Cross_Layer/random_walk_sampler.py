import networkx as nx
import random
import warnings
from collections import deque

def random_walk_sampler(Gs, node_exit, sample_size, nodeinfluence, metropolized=False, initial_node=None):
    seqnode = [initial_node]
    # 初始化当前节点
    current_node = initial_node
    curlayer = random.choice(node_exit[current_node])
    G = Gs[curlayer]
    visited_layer = deque(maxlen=3)
    # 设置固定长度的队列，先入先出
    while True:
        if len(seqnode) < sample_size:
                node_before_step = current_node
                current_node = next_node(G, current_node, metropolized)
                seqnode.append(current_node)
                if 1 - nodeinfluence[node_before_step] > random.uniform(0,1):
                    choices_layers = list(set(node_exit[node_before_step]) - set(visited_layer))
                    if choices_layers:
                        curlayer = random.choice(choices_layers)
                        if current_node not in Gs[curlayer].nodes():
                            current_node = initial_node
                        visited_layer.append(curlayer)
        else:
            break
    return seqnode


def next_node(G, current_node, metropolized):
    if metropolized:
        if list(G.neighbors(current_node)):
            candidate = random.choice(list(G.neighbors(current_node)))
            current_node = candidate if (random.random() < float(G.degree(current_node))/G.degree(candidate)) else current_node
    else:
        current_node = random.choice(list(G.neighbors(current_node)))
    return current_node


def ignore_initial_steps(G, metropolized, excluded_initial_steps, current_node):
    for _ in range(0, excluded_initial_steps):
        current_node = next_node(G, current_node, metropolized)
    return current_node

