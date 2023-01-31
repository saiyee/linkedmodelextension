import json
import os

import torch

import config
from config import Environment as env
from modelextension import model_extender
from util.models import get_ontology

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PredictionModels:
    classes_mapping = {}
    predicates_mapping = {}
    model: any
    embed: any


prediction_models: PredictionModels


def class_id(class_name: str, classes_mapping: {}):
    for object, i in classes_mapping.items():
        if class_name == object:
            return i


def tuple_in_ontology(tuple: (str, str, str)) -> bool:
    if tuple in get_ontology():
        return True

    return False


def init(model: config.Dataset = env.DEFAULT_DATASET):
    global prediction_models
    prediction_models = PredictionModels()
    # read classes mapping created by train.py
    with open(model.CLASSES_MAPPING_PATH, 'r') as file:
        prediction_models.classes_mapping = json.load(file)

    if env.LOG_LEVEL > 20:
        print('classes mapping: ')
        print(json.dumps(prediction_models.classes_mapping, indent=4))

    # read predicates mapping created by train.py
    with open(model.PREDICATES_MAPPING_PATH, 'r') as file:
        prediction_models.predicates_mapping = json.load(file)

    if env.LOG_LEVEL > 20:
        print('predicates mapping: ')
        print(json.dumps(prediction_models.predicates_mapping, indent=4))

    prediction_models.model = torch.load(model.TORCH_LP_MODEL_PATH)
    prediction_models.embed = torch.load(model.TORCH_LP_EMBEDDING_PATH)


def recommend(positives: [str],
              target_node: str,
              candidate_nodes: [str] = None,
              filter_invalid: bool = False,
              top_n: int = 10):
    global concept_prediction_models
    try:
        prediction_models
    except NameError:
        init()

    if not class_id(target_node, prediction_models.classes_mapping):
        return []  # if the target node is not known, return empty results

    # get most similar target nodes based on the result from node2vec if not preset
    if not candidate_nodes:
        positives = list(set(positives))  # remove duplicates
        positives = [x for x in positives if class_id(x, prediction_models.classes_mapping)]
        candidate_nodes = model_extender._recommend(positives=positives)
        candidate_nodes = candidate_nodes + positives  # add the positives back in, they are always a target

    candidate_relations = []
    s_id = class_id(target_node, prediction_models.classes_mapping)
    assert s_id is not None
    for node in candidate_nodes:
        o_id = class_id(node, prediction_models.classes_mapping)
        assert o_id is not None
        for pred, i in prediction_models.predicates_mapping.items():
            embed = prediction_models.embed
            w = prediction_models.model.w_relation
            emb_triplet = embed[s_id] * w[i] * embed
            scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
            scores, indices = torch.sort(scores, descending=True)
            rank = int((indices == o_id).nonzero())
            score = scores[rank]

            #subj = prediction_models.embed[s_id]
            #rel = prediction_models.model.w_relation[i]
            #obj = prediction_models.embed[o_id]
            #score = torch.sum(subj * rel * obj)
            candidate_relations.append(((str(target_node), str(pred), str(node)), score))

    top_list = sorted(candidate_relations, key=lambda item: item[1], reverse=True)

    result_list = []
    if filter_invalid:
        for tuple in top_list:
            if tuple_in_ontology((tuple[0], tuple[1], tuple[2])):
                result_list.append(tuple)
    else:
        result_list = top_list

    if top_n:
        result_list = result_list[:top_n]

    if env.LOG_LEVEL > 20:
        print("results:", result_list)

    return result_list
