import preprocess.modify_relations as modrel
import random
import numpy as np
from util.parse import encode_triples


def prepare_data(models, c_map=None, p_map=None, shuffle=False, multiply=1,
                 triple_weights=None, verbose=False, generate_test=False, inclusive=True):
    triples = []
    for model in models[:]:
        for triple in model:
            for mult in range(multiply):
                triples.append(tuple(triple))

    if shuffle:
        print("shuffling triples")
        random.shuffle(triples)

    if triple_weights:
        print("applying triple weights")
        triples = modrel.reduce_relations(triples, triple_weights, verbose=verbose)
    # print("training triples size", len(train_triples))

    encoded_triples = encode_triples(triples, c_map, p_map)

    if generate_test:
        if inclusive:
            train = encoded_triples
            mask = np.random.choice(train.shape[0], int(len(encoded_triples) / 10), replace=False)
            test = train[mask]
        else:
            cut = int(len(encoded_triples) / 10)
            train = encoded_triples[:len(encoded_triples)-cut]
            test = encoded_triples[len(encoded_triples)-cut:]

        X = train[:, [0, 2]]
        y = train[:, 1]

        X_test = test[:, [0, 2]]
        y_test = test[:, 1]
        return X, y, X_test, y_test

    train = encoded_triples
    X = train[:, [0, 2]]
    y = train[:, 1]
    # print(X.shape, y.shape)
    # print(train_data, X,y)
    return X, y
