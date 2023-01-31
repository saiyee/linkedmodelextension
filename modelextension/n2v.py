import os
from pathlib import Path

import networkx as nx
from node2vec import Node2Vec

import config as config
from config import Environment as env
from util.models import load_ontology

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def node2vec_ontology(model: config.Dataset = env.DEFAULT_DATASET):
    ontology = load_ontology(model.ONTOLOGY_PATH)
    # rdf_graph = rdflib.Graph()
    # rdf_graph.parse(ONTOLOGY_PATH, format='turtle')
    # ontology = []
    # for edge in rdf_graph.triples((None, None, None)):
    #    ontology.append(edge)
    graph = nx.Graph()
    for (s, a, t) in ontology:
        graph.add_edge(s, t)

    ontology_based_node2vec = Node2Vec(graph,
                                       p=1,
                                       q=1,
                                       dimensions=20,
                                       walk_length=3,
                                       num_walks=200,
                                       workers=2)

    ontology_based_model = ontology_based_node2vec.fit(window=5,
                                                       min_count=1)

    ontology_based_model.save(model.WORD2VEC_MODEL_PATH)


def node2vec_models(triples: [(str, str, str)],
                    ne_config: config.NodeEmbeddings = config.NodeEmbeddings,
                    model: config.Dataset = env.DEFAULT_DATASET):
    sums = {}
    weights = {}
    outgoing = {}

    for (s, a, t) in triples:
        if not (s, a, t) in weights:
            weights[(s, a, t)] = 0
        weights[(s, a, t)] += 1
        if not (s, t) in sums:
            sums[(s, t)] = 0
        if not s in outgoing:
            outgoing[s] = 0
        outgoing[s] += 1
        sums[(s, t)] += 1

    #weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))
    sorted_sums = sorted(sums.items(), reverse=True, key=lambda x: x[1])
    print(sorted_sums)

    total = sum([count for count in sums.values()])

    graph = nx.Graph()
    for (s, t), count in sums.items():
        weight = sums[(s, t)] / outgoing[s]
        graph.add_edge(str(s), str(t), weight=weight)

    model_based_node2vec = Node2Vec(graph,
                                    p=ne_config.p,
                                    q=ne_config.q,
                                    dimensions=ne_config.dimension,
                                    walk_length=ne_config.walk_length,
                                    num_walks=ne_config.walks,
                                    workers=ne_config.workers)
    print("walks size", len(model_based_node2vec.walks), model_based_node2vec.num_walks)

    model_based_model = model_based_node2vec.fit(window=ne_config.window, sg=1, negative=ne_config.negative,
                                                 ns_exponent=ne_config.ns_exponent)
    model_based_model.save(model.WORD2VEC_MODEL_PATH)


def node2vec(triples: [(str, str, str)], ne_config: config.NodeEmbeddings = config.NodeEmbeddings,
             model: config.Dataset = env.DEFAULT_DATASET):
    Path(model.WORD2VEC_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    if env.NODE2VEC_SELECT == 0:  # ontology based
        print('training', model.identifier, 'Node2Vec on ONTOLOGY')
        node2vec_ontology()
    elif env.NODE2VEC_SELECT == 1:  # model based
        print('training', model.identifier, 'Node2Vec on MODELS')
        node2vec_models(triples, ne_config)
