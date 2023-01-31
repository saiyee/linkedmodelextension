import json
import random
from datetime import datetime
from math import floor

import config
from config import Environment as env, GNN as gnn_config
from linkprediction.evaluation.evaluate_lp_me import evaluate_model
from linkprediction.train import train_models


def gnn_grid_search(model: config.Dataset = env.DEFAULT_DATASET):
    with open(model.PARSED_MODELS_PATH, 'r') as file:
        models = json.load(file)
        file.close()

    triples = [tup for mod in models for tup in mod]

    print("models triples size", len(triples))

    test_triples = random.sample(triples, floor(len(triples) / 10))

    all_results = []
    for learning_rate in [1e-3, 1e-4]:
        for no_epochs in [3000, 5000]:
            for no_negative_samples in [0, 5, 10, 20]:
                for graph_split_size in [0.5, 0.7]:
                    config = gnn_config()
                    config.LEARNING_RATE = learning_rate
                    config.N_EPOCHS = no_epochs
                    config.NEGATIVE_SAMPLE = no_negative_samples
                    config.GRAPH_SPLIT_SIZE = graph_split_size

                    results = train_models(train_triples=triples, test_triples=test_triples, config=config)
                    results['mrr'] = str(results['mrr'])
                    results['dataset'] = model.identifier
                    results['no_epochs'] = config.N_EPOCHS
                    results['learning_rate'] = config.LEARNING_RATE

                    recommendation_results = evaluate_model()
                    results['accuracy'] = recommendation_results['accuracy']
                    results['results'] = recommendation_results['result']

                    # current_results.append(results)
                    all_results.append(results)
                    print(json.dumps(results, indent=2, sort_keys=True))
            # with open("results/" + (datetime.now().strftime("%Y%m%d-%H%M") + "_all_results_"
            #                         + str(learning_rate) + "_" + str(no_epochs) + "_" + str(no_negative_samples) + ".json"),
            #           'w') as outfile:
            #     json.dump(current_results, outfile, indent=2, sort_keys=True)
    print(json.dumps(all_results, indent=2, sort_keys=True))
    with open("results/" + model.identifier + "_"
              + (datetime.now().strftime("%Y%m%d-%H%M") + "_all_results.json"),
              'w') as outfile:
        json.dump(all_results, outfile, indent=2, sort_keys=True)


if __name__ == '__main__':
    gnn_grid_search()
