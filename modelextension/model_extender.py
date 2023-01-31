import logging

import networkx as nx
from gensim.models import Word2Vec

from config import Environment as env, Dataset
from modelextension.statistics_recommender import StatisticsRecommender


class NodeEmbeddings:
    node2vec_model: Word2Vec
    statistic_recommender: StatisticsRecommender
    pecan: Word2Vec


concept_prediction_models: NodeEmbeddings


def class_id(class_name: str, classes_mapping: {}):
    for object, i in classes_mapping.items():
        if class_name == object:
            return i


def init(model: Dataset = env.DEFAULT_DATASET):
    global concept_prediction_models
    concept_prediction_models = NodeEmbeddings()

    concept_prediction_models.node2vec_model = Word2Vec.load(model.WORD2VEC_MODEL_PATH)
    #concept_prediction_models.statistic_recommender = StatisticsRecommender.load(model.SR_MODEL_PATH)


def _recommend(positives: [str], topn: int = 5) -> [(str, float)]:
    global concept_prediction_models
    try:
        concept_prediction_models
    except NameError:
        init()

    positives = list(set(positives))  # remove duplicates

    try:
        n2v_most_similar = concept_prediction_models.node2vec_model.wv.most_similar(
            topn=topn,
            positive=positives)
    except:
        logging.warning(f"Failed to retrieve n2v most similar for positives: {positives}")
        n2v_most_similar = []
        return n2v_most_similar

    # get most similar target nodes based on the result from node2vec
    most_similar_target_nodes = []

    logging.debug("n2v most similar")
    for node, probability in n2v_most_similar:
        logging.debug(f"\t{node} {probability}")
        most_similar_target_nodes = n2v_most_similar

    logging.debug(f'most similar target nodes: {most_similar_target_nodes}')

    return most_similar_target_nodes


def recommend_focus(history: [str], topn: int = 5) -> [str]:
    most_similar_target_nodes = _recommend(history,topn)

    return [x[0] for x in most_similar_target_nodes]  # drop the probabilities (for now)


def recommend_general(model: nx.Graph, topn: int = 5) -> [str]:
    aggregate = {}
    candidates = []
    for node in model.nodes:
        neighborhood = [str(x) for x in model.neighbors(node)]  # 1 hop neighbors
        filter_nodes = model.nodes
        most_similar_target_nodes = _recommend([str(node)] + neighborhood, topn)
        most_similar_target_nodes = [uri for uri in most_similar_target_nodes if uri[0] not in filter_nodes]
        # most_similar_target_nodes = most_similar_target_nodes[:1]
        logging.debug(f"{node} -> {most_similar_target_nodes}")
        #for uri, certainty in most_similar_target_nodes:
        #    if uri not in aggregate:
        #        aggregate[uri] = 0
        #    aggregate[uri] += certainty
        candidates.append({
            'node': node,
            'mstn': most_similar_target_nodes,
            'cert': sum([c for n,c in most_similar_target_nodes]) / len(most_similar_target_nodes) if len(most_similar_target_nodes) > 0 else 1

        })

    candidates = sorted(candidates, key=lambda x:x['cert'], reverse=True)
    candidates_max = {}
    for candidate in candidates:
        for uri, certainty in candidate['mstn']:
            if not uri in candidates_max:
                candidates_max[uri] = certainty
            else:
                candidates_max[uri] = max(candidates_max[uri], certainty)
            if uri not in aggregate:
                aggregate[uri] = 0
            aggregate[uri] += certainty

    # aggregate = sorted(aggregate.items(), key=lambda x: x[1], reverse=True)
    aggregate = sorted(candidates_max.items(), key=lambda x: x[1], reverse=True)

    return aggregate[:topn]
