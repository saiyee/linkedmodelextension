import json
from typing import Dict

import config
from config import Environment as env
from evaluate.EvaluationConfig import EvaluationConfig, intersection
from modelextension import model_extender


def evaluate_model(evaluation_config):
    if not isinstance(evaluation_config, EvaluationConfig):
        evaluation_config = EvaluationConfig(**evaluation_config)

    given_positives = evaluation_config.positives
    target_node = evaluation_config.target
    given_target_nodes = evaluation_config.target_nodes
    given_prediction_triplets = evaluation_config.prediction_triplets

    print("positive:", given_positives)
    result_target_nodes = model_extender.recommend_focus(history=given_positives)

    # print(result_target_nodes)
    found_target_nodes = intersection(given_target_nodes, result_target_nodes)

    print("Found target nodes:", found_target_nodes)

    me_accuracy = (len(found_target_nodes) / len(given_target_nodes))
    print("Accuracy (model_extension):", len(found_target_nodes), "/", len(given_target_nodes),
          me_accuracy)

    print("------------------------------------------------")

    return {
        'positives': given_positives,
        'target': target_node,
        # 'result': result_list,
        'targets': len(given_prediction_triplets),
        'matches': found_target_nodes,
        'accuracy': me_accuracy
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
        sum_found += len(result['matches'])
        print(json.dumps(result, indent=4))

    accuracy = sum_found / sum_expected
    print('Overall accuracy (model extension):', accuracy, '\n')

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
    print('Overall accuracy (model extension):', result['accuracy'])


if __name__ == '__main__':
    evaluate_all()
