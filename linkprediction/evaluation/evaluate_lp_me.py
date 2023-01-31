import json
from typing import Dict

import config
from config import Environment as env
from evaluate.EvaluationConfig import EvaluationConfig, intersection
from linkprediction import link_predictor
from modelextension import model_extender


def evaluate_model(evaluation_config):
    if not isinstance(evaluation_config, EvaluationConfig):
        evaluation_config = EvaluationConfig(**evaluation_config)

    given_positives = evaluation_config.positives
    target_node = evaluation_config.target
    given_target_nodes = evaluation_config.target_nodes
    given_prediction_triplets = evaluation_config.prediction_triplets

    result_list = link_predictor.recommend(given_positives, target_node)

    print("positive:", given_positives)

    print("results:")
    for result in result_list:
        print(result)

    # TODO use context here
    result_target_nodes = model_extender._recommend(positives=given_positives)
    print(result_target_nodes)
    found_target_nodes = intersection(given_target_nodes, result_target_nodes)

    print("Accuracy (model_extension):", len(found_target_nodes), "/", len(given_target_nodes),
          (len(found_target_nodes) / len(given_target_nodes)))

    results_triplets = [x[0] for x in result_list]
    found_triplets = intersection(results_triplets, given_prediction_triplets)

    print("Accuracy (linkprediction):", len(found_triplets), "/", len(given_prediction_triplets),
          (len(found_triplets) / len(given_prediction_triplets)))

    print("------------------------------------------------")

    return {
        'positives': given_positives,
        'target': target_node,
        # 'result': result_list,
        'targets': len(given_prediction_triplets),
        'matches': len(found_triplets),
        'accuracy': len(found_triplets) / len(given_prediction_triplets)
    }


def evaluate_all(model: config.Dataset = env.DEFAULT_DATASET):
    evaluation_configs: Dict[str, EvaluationConfig]
    with open(model.EVALUATION_CONFIG_PATH, 'r') as file:
        evaluation_configs = json.load(file)
    file.close()

    results = []
    for config in evaluation_configs.values():
        result = evaluate_model(config)
        results.append(result)

    sum_expected = 0
    sum_found = 0
    for result in results:
        sum_expected += result['targets']
        sum_found += result['matches']
        print(json.dumps(result, indent=4))

    accuracy = sum_found / sum_expected
    print('Overall accuracy (linkprediction):', accuracy, '\n')

    return accuracy


def evaluate_single(model: config.Dataset = env.DEFAULT_DATASET, evaluation_config_identifier: str = None):
    evaluation_configs: Dict[str, EvaluationConfig]
    with open(model.EVALUATION_CONFIG_PATH, 'r') as file:
        evaluation_configs = json.load(file)

    if evaluation_config_identifier:
        evaluation_config = evaluation_configs[evaluation_config_identifier]
    else:
        evaluation_config = list(filter(lambda x: x["default"], evaluation_configs.values()))[0]

    result = evaluate_model(evaluation_config)
    print('Overall accuracy (linkprediction):', result['accuracy'])


if __name__ == '__main__':
    evaluate_all()
