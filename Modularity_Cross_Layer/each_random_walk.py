import networkx as nx
import random
import warnings


def random_walk_sampler(G, sample_size, initial_node=None, metropolized=False, excluded_initial_steps=0):
    seqnode = [initial_node]
    path = []
    # 设置参数验证
    if type(G) != nx.Graph:
        raise nx.NetworkXException("Graph must be a simple undirected graph!")
    if initial_node is not None and G.has_node(initial_node):
        current_node = initial_node
    else:
        current_node = random.choice(G.nodes())
        if initial_node is not None:
            warnings.warn('Initial node could not be found in population graph. It was chosen randomly.')

    # 初始化当前节点
    current_node = ignore_initial_steps(G, metropolized, excluded_initial_steps, current_node)
    while True:
        if len(seqnode) < sample_size:
            node_before_step = current_node
            count = 0
            while node_before_step == current_node:
                current_node = next_node(G, current_node, metropolized)
                count += 1
                if count > 5:
                    current_node = random.choice(list(G.nodes()))
            seqnode.append(current_node)
            path.extend([node_before_step, current_node])
        else:
            break
    return seqnode


def next_node(G, current_node, metropolized):
    if metropolized:
        if list(G.neighbors(current_node)):
            candidate = random.choice(list(G.neighbors(current_node)))
            current_node = candidate if (random.random() < float(G.degree(current_node))/G.degree(candidate)) else current_node
    return current_node


def ignore_initial_steps(G, metropolized, excluded_initial_steps, current_node):
    for _ in range(0, excluded_initial_steps):
        current_node = next_node(G, current_node, metropolized)
    return current_node

