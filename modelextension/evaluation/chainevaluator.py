import collections
import random

import networkx as nx

from modelextension import model_extender


class ChainEvaluator:

    def __init__(self, gold_triplets: [(str, str, str)]):
        self.id = id
        self.triplets = gold_triplets
        self.gold_nodes = set([x for (x, p, o) in gold_triplets])
        self.gold_nodes = self.gold_nodes.union(set([x for (s, p, x) in gold_triplets]))

    def _get_unmapped_random_node(self, current_nodes: [str]):
        unmapped_nodes = [x for x in self.gold_nodes if x not in current_nodes]
        if len(unmapped_nodes) == 0:
            return None
        random_node = random.sample(unmapped_nodes, 1)[0]
        return random_node

    def _build_chain(self, chain: [str], model_nodes: [str], current_node: str, history: collections.deque):
        history.append(current_node)
        chain.append(current_node)
        model_nodes.append(current_node)
        local_recommended = model_extender.recommend_focus(history=history)
        local_recommended = [concept for concept in local_recommended if concept not in model_nodes]
        graph = nx.Graph()
        graph.add_nodes_from(nodes_for_adding=model_nodes)
        general_recommend = model_extender.recommend_general(model=graph)
        recommendations = local_recommended[:4] + general_recommend[:1]
        matches = [r for r in recommendations if r in self.gold_nodes]
        if len(matches) == 0:
            return chain, model_nodes, current_node, history
        else:
            results = []
            for match in matches:
                results.append(self._build_chain([x for x in chain], [x for x in model_nodes], match, history.copy()))
        results = sorted(results, key=lambda x: len(x[0]), reverse=True)  # todo check multiple top length chains
        return list(results)[0]

    def evaluate(self):
        random_node = self._get_unmapped_random_node([])
        model_nodes = []
        current_node = random_node
        trail = []
        chain = []
        history = collections.deque(maxlen=10)
        while len(model_nodes) < len(self.gold_nodes):
            chain, model_nodes, current_node, history = self._build_chain(chain=chain, model_nodes=model_nodes,
                                                                          current_node=current_node, history=history)
            trail.append(chain)
            history.clear()

            # no more matches, find new node
            current_node = self._get_unmapped_random_node(model_nodes)

            chain = []
        return trail

    def get_nodes_of_model(self, model: [(str, str, str)]):
        nodes = set([x if x else y for (x, p, o) in model for (s, p, y) in model])
        return nodes
