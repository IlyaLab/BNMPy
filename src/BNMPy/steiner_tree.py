# steiner tree implementation in graph, mehlhorn approximation
# based on the networkx implementation at https://networkx.org/documentation/stable/_modules/networkx/algorithms/approximation/steinertree.html#steiner_tree
import igraph as ig

def steiner_tree_wrapper(graph, ids, method='takahashi'):
    """
    A thin wrapper around a couple of approximate steiner tree algorithms.

    Args:
        graph - an igraph.Graph
        ids - a list of node names
        method - one of 'takahashi', 'mehlhorn'

    Returns:
        tree that contains all nodes in ids as leaves.
    """
    indices = [graph.vs.find(name=i).index for i in ids]
    if method == 'takahashi':
        tree = takahashi_matsuyama_steiner_tree(graph, indices)
    elif method == 'takahashi_multi_start':
        min_tree_size = len(graph.vs)
        best_tree = graph
        for i in range(len(indices)):
            tree = takahashi_matsuyama_steiner_tree(graph, indices, initial_terminal=i)
            if len(tree.vs) < min_tree_size:
                best_tree = tree
                min_tree_size = len(tree.vs)
        tree = best_tree
    elif method == 'shortest_paths':
        tree = shortest_paths_steiner_tree(graph, indices)
    elif method == 'mehlhorn':
        tree = mehlhorn_steiner_tree(graph, indices)
    else:
        raise ValueError('Error: method must be one of "takahashi", "takahashi_multi_start", "mehlhorn", or "shortest_paths"')
    ids_set = set(ids)
    for n in tree.vs:
        if n['name'] in ids_set:
            n['in_query'] = 1
        else:
            n['in_query'] = 0
    return tree


def steiner_tree(G, source_nodes, method='takahashi'):
    """
    Args:
        G - an igraph graph
        source_nodes - a list of node names (IDs)
        method - one of 'takahashi', 'mehlhorn', 'shortest_paths'

    Returns: a subgraph
    """
    indices = [G.vs.find(name=i).index for i in source_nodes]
    if method == 'takahashi':
        tree = takahashi_matsuyama_steiner_tree(G, indices)
    elif method == 'takahashi_multi_start':
        min_tree_size = len(G.vs)
        best_tree = None
        for i in range(len(source_nodes)):
            tree = takahashi_matsuyama_steiner_tree(G, indices, initial_terminal=i)
            if len(tree.vs) < min_tree_size:
                best_tree = tree
                min_tree_size = len(tree.vs)
        tree = best_tree
    elif method == 'shortest_paths':
        tree = shortest_paths_steiner_tree(G, indices)
    elif method == 'mehlhorn':
        tree = mehlhorn_steiner_tree(G, indices)
    else:
        raise ValueError('Error: method must be one of "takahashi", "takahashi_multi_start", "mehlhorn", or "shortest_paths"')
    return tree


def multi_source_shortest_paths(G, source_nodes, dest_nodes=None, shortest_paths_cache=None):
    """
    Returns the shortest paths from any of the nodes in source_nodes to every node in dest_nodes.

    Assumes that everything in source_nodes is an index into G.

    source_nodes are graph node indices.

    shortest_paths_cache is a dict containing the results of G.get_shortest_paths(n) - all-dest shortest paths for node n.

    Output:
        shortest-paths - dict of {node_index: [nodes from a source node]}
    """
    if shortest_paths_cache is None:
        shortest_paths_cache = {}
    shortest_paths = {}
    for n in source_nodes:
        if n in shortest_paths_cache:
            paths = shortest_paths_cache[n]
        else:
            paths = G.get_shortest_paths(n, dest_nodes)
            shortest_paths_cache[n] = paths
        shortest_paths[n] = paths
    return shortest_paths


def mehlhorn_steiner_tree(G, terminal_nodes):
    """
    Mehlhorn algorithm:

    1. Construct graph G1, where the nodes are terminal nodes and the distances between the nodes are the shortest path in G.
    2. Find a minimum spanning tree G2 of G1.
    3. Construct a subgraph G3 of G by replacing each edge in G2 by the corresponding minimal path.
    4. Find a minimum spanning tree G4 of G3.
    5. Construct a Steiner tree G5 from G4 by deleting edges and nodes from G4, if necessary, so that no leaves in G5 are steiner vertices (that is, remove all leaf nodes that aren't part of the terminal nodes).
    """
    all_terminals = set([G.vs[n]['name'] for n in terminal_nodes])
    # 1. find all source shortest paths from the terminal nodes
    paths = multi_source_shortest_paths(G, terminal_nodes, terminal_nodes)
    # 2. G1 - construct a complete graph containing all terminal nodes, with the edges as distances
    G1 = ig.Graph.Full(len(terminal_nodes))
    for v in G1.vs:
        v['name'] = G.vs[terminal_nodes[v.index]]['name']
    weights = []
    for edge in G1.es:
        path = paths[terminal_nodes[edge.source]][edge.target]
        weights.append(len(path) - 1)
        edge['weight'] = len(path) - 1
        edge['path'] = path
    # 3. G2 -  spanning tree of the complete graph
    G2 = G1.spanning_tree(weights)
    # G3 is a new graph that contains all shortest paths comprised by the weights
    G3 = ig.Graph()
    for edge in G2.es:
        prev_node = None
        for n in edge['path']:
            name = G.vs[n]['name']
            if len(G3.vs) == 0 or len(G3.vs.select(name=name)) == 0:
                G3.add_vertex(name)
            new_node = G3.vs.find(name=name)
            if prev_node is not None:
                # add edge
                G3.add_edge(prev_node.index, new_node.index)
            prev_node = new_node
    # G4 is the spanning tree of G3
    G4 = G3.spanning_tree()
    # prune leaves that don't belong to the terminal nodes
    has_nonterminal_leaf = True
    while has_nonterminal_leaf:
        has_nonterminal_leaf = False
        to_prune = []
        for v in G4.vs:
            if len(G4.neighbors(v.index))==1 and v['name'] not in all_terminals:
                to_prune.append(v.index)
                has_nonterminal_leaf = True
        if has_nonterminal_leaf:
            G4.delete_vertices(to_prune)
    return G4

def shortest_paths_steiner_tree(G, terminal_nodes):
    """
    This *might* be equivalent to the Takahashi method, but is somewhat more efficient.

    1. Find all-pairs shortest paths on G between all nodes in terminal_nodes.
    2. Create an induced subgraph containing all nodes on any of the shortest paths.
    3. Take a minimum spanning tree of the induced subgraph.
    4. Iteratively prune the MST to remove all leaves that are not terminal nodes.
    """
    all_terminals = set([G.vs[n]['name'] for n in terminal_nodes])
    # 1. find all source shortest paths from the terminal nodes
    paths = multi_source_shortest_paths(G, terminal_nodes, terminal_nodes)
    # create a MST on the subgraph containing all nodes on the shortest paths
    all_path_nodes = set()
    for _, shortest_paths in paths.items():
        for path in shortest_paths:
            all_path_nodes.update(path)
    subgraph = G.induced_subgraph(all_path_nodes)
    spanning_tree = subgraph.spanning_tree()
    # prune leaves that don't belong to the terminal nodes
    has_nonterminal_leaf = True
    while has_nonterminal_leaf:
        has_nonterminal_leaf = False
        to_prune = []
        for v in spanning_tree.vs:
            if len(spanning_tree.neighbors(v.index))==1 and v['name'] not in all_terminals:
                to_prune.append(v.index)
                has_nonterminal_leaf = True
        if has_nonterminal_leaf:
            spanning_tree.delete_vertices(to_prune)
    return spanning_tree


   
def takahashi_matsuyama_steiner_tree(G, terminal_nodes, initial_terminal=0):
    """
    Takahashi and Matsuyama algorithm: I can't find the paper so I'm working off the wikipedia description, and the description in https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-144

    1. start with one arbitrary terminal t
    2. find the terminal s closest to t, add the shortest path to the subgraph G'
    3. Find the closest terminal to any node in G', add that path to G'
    Continue until all nodes in G' have been added to the graph.
    Then return the minimum spanning tree of G', and remove all non-terminals with only one edge.

    According to https://arxiv.org/pdf/1409.8318v1.pdf, it produces smaller steiner trees (better approximation factor) than the Mehlhorn algorithm.
    """
    all_terminals = set([G.vs[n]['name'] for n in terminal_nodes])
    terminal_nodes = list(set(terminal_nodes.copy()))
    t = terminal_nodes[initial_terminal]
    terminal_nodes.pop(initial_terminal)
    # get the nearest node from the initial terminal
    subgraph_nodes = set()
    paths = G.get_shortest_paths(t, terminal_nodes)
    shortest_length = -1
    shortest_path = []
    nearest_terminal = 0
    terminal_paths_to_subgraph = {}
    for i, path in enumerate(paths):
        terminal_paths_to_subgraph[terminal_nodes[i]] = path
        length = len(path) - 1
        if length < shortest_length or shortest_length < 0:
            shortest_length = length
            nearest_terminal = i
            shortest_path = path
    terminal_nodes.pop(nearest_terminal)
    subgraph_nodes.update(shortest_path)
    new_subgraph_nodes = set(shortest_path[1:])
    # add path to G'
    while terminal_nodes:
        # get distances from terminal nodes to new_subgraph_nodes, store the shortest path from each terminal to the subgraph
        shortest_length = -1
        shortest_path = []
        nearest_terminal = 0
        for i, t in enumerate(terminal_nodes):
            paths = G.get_shortest_paths(t, new_subgraph_nodes)
            for _, path in enumerate(paths):
                length = len(path) - 1
                if length < len(terminal_paths_to_subgraph[t]):
                    terminal_paths_to_subgraph[t] = path
                if shortest_length < 0 or len(terminal_paths_to_subgraph[t]) < shortest_length:
                    shortest_length = len(terminal_paths_to_subgraph[t])
                    shortest_path = terminal_paths_to_subgraph[t]
                    nearest_terminal = i
        if (shortest_length < 0):
            print('Steiner tree approximation algorithm warning: no path from terminals found?')
            print('terminal_nodes:', terminal_nodes)
            print('new_subgraph_nodes:', new_subgraph_nodes)
        terminal_nodes.pop(nearest_terminal)
        subgraph_nodes.update(shortest_path)
        new_subgraph_nodes = set(shortest_path)
    subgraph = G.induced_subgraph(subgraph_nodes)
    # calculate minimal spanning tree from induced subgraph
    spanning_tree = subgraph.spanning_tree()
    # prune leaves that don't belong to the terminal nodes
    has_nonterminal_leaf = True
    while has_nonterminal_leaf:
        has_nonterminal_leaf = False
        to_prune = []
        for v in spanning_tree.vs:
            if len(spanning_tree.neighbors(v.index))==1 and v['name'] not in all_terminals:
                to_prune.append(v.index)
                has_nonterminal_leaf = True
        if has_nonterminal_leaf:
            spanning_tree.delete_vertices(to_prune)
    return spanning_tree
