import json
from typing import Dict

import config
from config import Environment as env
from evaluate.EvaluationConfig import EvaluationConfig, intersection
from evaluate.EvaluationResult import EvaluationResult
from linkprediction import link_predictor
from modelextension import model_extender


def evaluate(evaluation_config) -> EvaluationResult:
    if not isinstance(evaluation_config, EvaluationConfig):
        evaluation_config = EvaluationConfig(**evaluation_config)

    given_positives = evaluation_config.positives
    target_node = evaluation_config.target
    given_target_nodes = evaluation_config.target_nodes
    given_prediction_triplets = evaluation_config.prediction_triplets

    result_list = link_predictor.recommend(positives=given_positives,
                                           target_node=target_node,
                                           candidate_nodes=given_target_nodes,
                                           top_n=3)

    return EvaluationResult(identifier=evaluation_config.identifier,
                            node=target_node,
                            positives=given_positives,
                            targets=given_prediction_triplets,
                            result=result_list)


def evaluate_all(model: config.Dataset = env.DEFAULT_DATASET) -> float:
    evaluation_configs: Dict[str, EvaluationConfig]
    with open(model.EVALUATION_CONFIG_PATH, 'r') as file:
        evaluation_configs = json.load(file)
    file.close()

    results = []
    for config_identifier in evaluation_configs.keys():
        result = evaluate_single_config(model, config_identifier)
        results.append(result)

    accurracies = 0
    for result in results:
        accurracies += result.accuracy
        print(json.dumps(result, indent=4))

    accuracy = accurracies / len(results)
    print('Accuracy:', 'overall', accuracy, '\n')
    return accuracy


def evaluate_single_config(model: config.Dataset = env.DEFAULT_DATASET,
                           evaluation_config_identifier: str = None) -> EvaluationResult:
    evaluation_configs: Dict[str, EvaluationConfig]
    with open(model.EVALUATION_CONFIG_PATH, 'r') as file:
        evaluation_configs = json.load(file)

    if evaluation_config_identifier:
        evaluation_config = evaluation_configs[evaluation_config_identifier]
    else:
        evaluation_config = list(filter(lambda x: x["default"], evaluation_configs.values()))[0]

    result = evaluate(evaluation_config)
    return result


if __name__ == '__main__':
    evaluate_single_config(evaluation_config_identifier="Offer")
