import json
from datetime import datetime

import config
from config import Environment as env
from modelextension import n2v
from modelextension.evaluation.evaluate_me import evaluate_all


def node_embedding_grid_search(model: config.Dataset = env.DEFAULT_DATASET):
    with open(model.PARSED_MODELS_PATH, 'r') as file:
        models = json.load(file)
        file.close()

    triples = [tup for mod in models for tup in mod]

    all_results = []
    best: {} = {
        'accuracy': 0
    }
    for walks in [500, 1000, 1500, 2000]:
        for ns_exponent in [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]:
            #for q in [1, 2, 5, 10]:
                for window in [5, 10, 20]:
                    results = {}
                    #results['q'] = q
                    results['walks'] = walks
                    results['ns_exponent'] = ns_exponent
                    results['window'] = window
                    accuracies = []
                    for i in range(5):
                        #ne_config = config.NodeEmbeddings(q=q, walks=walks, ns_exponent=ns_exponent, window=window)
                        ne_config = config.NodeEmbeddings(walks=walks, ns_exponent=ns_exponent, window=window)
                        n2v.node2vec(triples, ne_config=ne_config)
                        accuracies.append(evaluate_all())
                    results['accuracy'] = sum(accuracies) / len(accuracies)
                    all_results.append(results)
                    if results['accuracy'] > best['accuracy']:
                        print('Found new best', results)
                        best = results

    print(json.dumps(all_results, indent=2, sort_keys=True))
    with open("results/" + model.identifier + (datetime.now().strftime("%Y%m%d-%H%M") + "_all_results.json"),
              'w') as outfile:
        json.dump(all_results, outfile, indent=2, sort_keys=True)


if __name__ == '__main__':
    node_embedding_grid_search()
