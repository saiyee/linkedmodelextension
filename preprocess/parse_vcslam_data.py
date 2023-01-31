import json
import random
from typing import Dict

from config import VCSLAM
from util.parse import get_graphs_from_rdf
from evaluate.EvaluationConfig import EvaluationConfig

from config import Environment as env


def parse_vcslam_data():
    selected_model = VCSLAM()

    evaluation_configs: Dict[str, EvaluationConfig]
    with open(selected_model.EVALUATION_CONFIG_PATH, 'r') as file:
        evaluation_configs = json.load(file)
    file.close()

    graphs = get_graphs_from_rdf(selected_model.MODELS_PATH)
    models = []

    for graph in graphs:
        model = []
        for (sub, pred, obj) in graph.triples((None, None, None)):
            model.append((str(sub), str(pred), str(obj)))
        models.append(model)

    prediction_triples = []
    for config in evaluation_configs.values():
        for triple in config['_prediction_triplets']:
            prediction_triples.append(triple)

    modified_models = []

    for model in models:
        modified_model = []
        for triple in model:
            triple = list(triple)
            add_triple = 0
            if triple in prediction_triples:
                add_triple = 1
            if add_triple == 0:
                continue
            modified_model.append(triple)
        modified_models.append(modified_model)

    with open(selected_model.PARSED_MODELS_PATH, 'w') as file:
        json.dump(modified_models, file, indent=4)
        file.close()


if __name__ == '__main__':
    parse_vcslam_data()
