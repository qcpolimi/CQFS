from collections import namedtuple
from math import floor

import dimod
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from dwave.system import DWaveSampler
from matplotlib.pyplot import figure

Coord = namedtuple('Coord', 'qubit cell row col')


def _get_nx_graph(graph):
    if not isinstance(graph, nx.Graph):
        return nx.Graph(graph)
    return graph


def get_nodes(graph):
    if isinstance(graph, nx.Graph):
        return sorted(list(graph.nodes))
    nodes_set = set()
    for e in graph:
        nodes_set.update(str(e))
    return list(nodes_set)


def get_nodes_from_embedding(embedding):
    nodes = set()
    for k, v in embedding.items():
        nodes.update(v)
    nodes = list(nodes)

    return nodes


def get_edges(graph):
    if isinstance(graph, nx.Graph):
        return list(graph.edges)
    return graph


def get_edges_from_bqm(bqm: dimod.BQM):
    return list(bqm.quadratic.keys())


def get_edges_from_embedding(graph: nx.Graph, embedding: dict):
    nodes = get_nodes_from_embedding(embedding)
    subgraph: nx.Graph = graph.subgraph(nodes)
    return list(subgraph.edges)


def get_graph_from_topology(qpu_topology):
    AVAILABLE_TOPOLOGIES = ['chimera', 'pegasus']

    assert qpu_topology in AVAILABLE_TOPOLOGIES, f"No graph available for the requested topology ({qpu_topology})." \
                                             f"Available topologies are {AVAILABLE_TOPOLOGIES}."

    if qpu_topology == 'pegasus':
        return dnx.chimera_graph(16)
    if qpu_topology == 'chimera':
        return dnx.pegasus_graph(16)


def get_identity_embedding(graph: nx.Graph):
    nodes = get_nodes(graph)
    return {str(n): [n] for n in nodes}


def _graph_hash(graph):
    graph = _get_nx_graph(graph)
    return nx.weisfeiler_lehman_graph_hash(graph)


def _get_coordinates(node_index):
    qubit_index = node_index % 8
    cell_index = floor(node_index / 8)
    row_index = floor(cell_index / 16)
    col_index = cell_index % 16

    return Coord(qubit_index, cell_index, row_index, col_index)


def _get_margins(nodes):
    coords = list(map(_get_coordinates, nodes))

    min_row = min(map(lambda x: x.row, coords))
    max_row = max(map(lambda x: x.row, coords))

    min_col = min(map(lambda x: x.col, coords))
    max_col = max(map(lambda x: x.col, coords))

    return min_row, max_row, min_col, max_col


def _get_node_size(nodes, min_size=80, max_size=300):
    min_row, max_row, min_col, max_col = _get_margins(nodes)
    n = max(max_row - min_row, max_col, min_col)

    a = min_size - max_size

    return a / 15 * (n - 1) + max_size


def _check_embedding_validity(embedding, qpu_topology):
    target_sampler = DWaveSampler(solver={'topology__type': qpu_topology})
    target_nodelist = np.array(target_sampler.nodelist)

    embedding_nodelist = np.array(get_nodes_from_embedding(embedding))
    return np.alltrue(np.isin(embedding_nodelist, target_nodelist))


def fully_connected_graph_from_size(size):
    edgelist = []
    for i in range(size):
        for j in range(size):
            edgelist.append((i, j))
    return edgelist


def draw_embedding_from_nodes(nodes, qpu_graph='pegasus', active_nodes=None, full_drawing=True, setup_figure=True):
    assert qpu_graph in ['pegasus', 'chimera'], \
        "The chosen QPU graph is not valid. Please choose one between pegasus and chìmera."

    if setup_figure:
        figure(num=None, figsize=(32, 32), dpi=80, facecolor='w', edgecolor='k')

    if qpu_graph == 'pegasus':
        graph = dnx.pegasus_graph(16, node_list=nodes)
    elif qpu_graph == 'chimera':
        graph = dnx.chimera_graph(16, node_list=nodes)
    else:
        return

    if full_drawing:
        node_size = 80
        width = 1.5
        edge_color = 'c'
        base_color = '#999999'

        if qpu_graph == 'pegasus':
            full_pegasus = dnx.pegasus_graph(16, node_list=active_nodes)
            dnx.draw_pegasus(full_pegasus, node_color=base_color, node_size=node_size / 2, edge_color=base_color)
        elif qpu_graph == 'chimera':
            full_chimera = dnx.chimera_graph(16, node_list=active_nodes)
            dnx.draw_chimera(full_chimera, node_color=base_color, node_size=node_size / 2, edge_color=base_color)
    else:
        node_size = _get_node_size(nodes)
        width = 1
        edge_color = 'k'

    if qpu_graph == 'pegasus':
        dnx.draw_pegasus(graph, node_size=node_size, edge_color=base_color, width=width)
    elif qpu_graph == 'chimera':
        dnx.draw_chimera(graph, node_size=node_size, edge_color=base_color, width=width)


def draw_embedding(embedding, qpu_graph='pegasus', active_nodes=None, highlight_variables=None, full_drawing=True,
                   show=True):
    assert qpu_graph in ['pegasus', 'chimera'], \
        "The chosen QPU graph is not valid. Please choose one between pegasus and chìmera."

    if qpu_graph == 'pegasus':
        figure(num=None, figsize=(48, 48), dpi=80, facecolor='w', edgecolor='k')
    elif qpu_graph == 'chimera':
        figure(num=None, figsize=(32, 32), dpi=80, facecolor='w', edgecolor='k')

    nodes = get_nodes_from_embedding(embedding)

    draw_embedding_from_nodes(nodes, qpu_graph=qpu_graph, active_nodes=active_nodes, full_drawing=full_drawing,
                              setup_figure=False)

    if highlight_variables is not None:

        if full_drawing:
            node_size = 100
        else:
            node_size = _get_node_size(nodes, min_size=100)

        highlight_nodes = set()
        for var in highlight_variables:
            try:
                highlight_nodes.update(embedding[var])
            except KeyError:
                print("No variable {} in the embedding!".format(var))

        if qpu_graph == 'pegasus':
            highlight_pegasus = dnx.pegasus_graph(16, node_list=highlight_nodes)
            dnx.draw_pegasus(highlight_pegasus, node_color='r', node_size=node_size, edge_color='r', width=2)
        elif qpu_graph == 'chimera':
            highlight_chimera = dnx.chimera_graph(16, node_list=highlight_nodes)
            dnx.draw_chimera(highlight_chimera, node_color='r', node_size=node_size, edge_color='r', width=2)

    if show:
        plt.show()
