import json
from typing import Dict

import config
from config import Environment as env
from evaluate.EvaluationConfig import EvaluationConfig, intersection
from linkprediction import link_predictor
from modelextension import model_extender
from preprocess.steiner_tree import compute_steiner_tree, get_target_nodes
import random
from math import floor
from config import GNN as gnn_config
from linkprediction.evaluation.evaluate_lp_me import evaluate_all
from linkprediction.train import train_models
from modelextension import statistics_recommender, n2v


def identify_concepts(index: int, model: config.Dataset = env.DEFAULT_DATASET):
    with open(model.PARSED_MODELS_PATH, 'r') as file:
        models = json.load(file)

    target_triplets = []

    steiner_tree = compute_steiner_tree(index,model)

    candidate_nodes = get_target_nodes(index,model)

    for s, p, o in models[index]:
        if s in steiner_tree and o in candidate_nodes:
            target_triplets.append((s,p,o))

    return target_triplets


def evaluate_models(indexes: [], model: config.Dataset = env.DEFAULT_DATASET):

    ranks = []
    total_targets = 0
    checked_triplets = []

    for index in indexes:
        targets = identify_concepts(index=index, model=model)
        targets = [target for target in targets if target not in checked_triplets]
        checked_triplets = list(set(checked_triplets + targets))
        candidate_nodes = list(set([o for s,p,o in targets]))
        total_targets += len(targets)
        for target in targets:
            anchor = target[0]
            result_list = link_predictor.recommend_by_anchors(anchor=anchor, candidate_nodes=candidate_nodes)
            result_list = [t for t,r in result_list]
            print("results:", anchor)
            for result in result_list:
                print(result)
            try:
                rank = result_list.index(target)
                ranks.append(rank)
            except ValueError:
                rank = 5
                ranks.append(rank)


    # print("Accuracy (linkprediction):", len(found_triplets), "/", len(given_prediction_triplets),
    #       (len(found_triplets) / len(given_prediction_triplets)))

    print("------------------------------------------------")
    print("hits@1", len([rank for rank in ranks if rank == 0]), len([rank for rank in ranks if rank == 0])/total_targets)
    print("hits@3", len([rank for rank in ranks if rank <= 2]),len([rank for rank in ranks if rank <= 2])/total_targets)
    print("hits@5", len([rank for rank in ranks if rank <= 4]),len([rank for rank in ranks if rank <= 4])/total_targets)
    print("hits@10", len([rank for rank in ranks if rank <= 9]),len([rank for rank in ranks if rank <= 9])/total_targets)

if __name__ == '__main__':
    evaluate_models(indexes=[12,13,14,15,16,17])
