import collections
import logging
import random

import networkx as nx

from modelextension import model_extender
from preprocess.steiner_tree import get_target_nodes


def get_nodes_of_model(model: [(str, str, str)]):
    nodes = set([x if x else y for (x, p, o) in model for (s, p, y) in model])
    return nodes


class GeneralEvaluator:

    def __init__(self, triplets, target_nodes, initial_graph: nx.Graph):
        logging.debug(f"input triplets:{triplets}")
        self.triplets = triplets
        logging.debug(f"input target nodes: {target_nodes}")
        self.target_nodes = target_nodes
        logging.debug(f"input initial graph: {initial_graph.nodes}")
        self.initial_graph = initial_graph

    def _get_unmapped_random_node(self, current_nodes: [str]):
        unmapped_nodes = [x for x in self.target_nodes if x not in current_nodes]
        if len(unmapped_nodes) == 0:
            return None
        random_node = random.sample(unmapped_nodes, 1)[0]
        return random_node

    def evaluate(self):
        model_graph = self.initial_graph.copy()
        recommended_nodes = set()
        while len(model_graph) < len(self.initial_graph) + len(self.target_nodes):
            recommendations = model_extender.recommend_general(model=model_graph)
            # print(recommendations)
            matches = [x[0] for x in recommendations if x[0] in self.target_nodes and x[0] not in model_graph.nodes]
            if len(matches) == 0:

                # no more matches, find new node
                current_node = self._get_unmapped_random_node(model_graph)
                if current_node is None:
                    break
                logging.debug(f"no matches found, adding {current_node} to model")
                model_graph.add_node(current_node)
            else:
                logging.debug(f"found matches {matches}")
                model_graph.add_nodes_from(matches)
                recommended_nodes |=  set(matches)

        return recommended_nodes
