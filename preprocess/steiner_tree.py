import json
from pathlib import Path
from typing import Dict, List, Any

import rdflib
from networkx import minimum_spanning_tree
import networkx as nx
from networkx.algorithms.approximation import steiner_tree

import config
from config import Environment as env
from visualization.visualize import visualize


def get_anchored_target_nodes(model: [], mapped_nodes: [], filter_target_nodes=True) -> dict[Any, list[Any]]:

    graph = nx.MultiDiGraph()
    for s, p, o in model:
        graph.add_edge(s, o, relation=str(p))

    minimal_tree = compute_steiner_tree(model, mapped_nodes)

    all_nodes = set(graph.nodes)
    tree_nodes = set(minimal_tree.nodes)
    target_nodes = all_nodes.difference(tree_nodes)

    anchored_nodes = {}
    # find the anchor for each target node
    for anchor in graph.nodes:
        if filter_target_nodes and anchor in target_nodes:
            continue
        found_nodes = []
        for target in target_nodes:
            if graph.has_edge(anchor,target):
                relation = graph.edges[0][anchor][target]['relation']
                found_nodes.append((target,relation))

        if found_nodes:
            anchored_nodes[anchor] = found_nodes

    return anchored_nodes

def get_target_nodes(index: int, model: config.Dataset = env.DEFAULT_DATASET) -> set[str]:

    with open(model.PARSED_MODELS_PATH, 'r') as file:
        models = json.load(file)

    graph = nx.Graph()
    for s, p, o in models[index]:
        graph.add_edge(s, o)

    minimal_tree = compute_steiner_tree(index, model)

    all_nodes = set(graph.nodes)
    tree_nodes = set(minimal_tree.nodes)
    target_nodes = all_nodes.difference(tree_nodes)
    return target_nodes


def compute_steiner_tree(model, mapped_nodes) -> nx.Graph:
    graph = nx.Graph()
    for s, p, o in model:
        if s == o:
            continue
        graph.add_node(s)
        graph.add_node(o)
        graph.add_edge(s, o)

    mst = minimum_spanning_tree(graph, algorithm='prim')
    # visualize(mst)

    minimal_tree = steiner_tree(mst, list(mapped_nodes))

    return minimal_tree


def get_mapped_attributes(index: int, model: config.Dataset = env.DEFAULT_DATASET) -> set[str]:
    base_path = Path(model.LABEL_MAPPINGS_PATH)
    mapping_files = list(base_path.glob('*'))
    with open(mapping_files[index], encoding='utf-8') as mappings_file:
        mappings = json.load(mappings_file)

    label_mappings= [x['conceptResource'] for x in mappings]
    label_mappings = set([x.replace('http://www.plasma.uni-wuppertal.de/schema#', 'http://tmdtkg#') for x in label_mappings])
    return label_mappings
