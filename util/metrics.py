from typing import List, Tuple
import numpy as np


def rank(recommendations: List[Tuple[Tuple[str, str,str], float]], valids: List[List[str]]):
    valids = [(s,p,o) for s,p,o in valids]
    recommendations = [item[0] for item in recommendations]
    indexes = [i for i in range(len(recommendations)) if recommendations[i] in valids]
    if not indexes:
        return 100
    #print(indexes)
    return min(indexes) +1


def calc_hits_mrr(predictions, reference, hits=None):
    if hits is None:
        hits = [1,3]
    predictions = np.argsort(predictions)[:, ::-1]
    positions = []
    for row in range(len(predictions)):
        for i in range(predictions.shape[1]):
            if predictions[row][i] == reference[row]:
                positions.append(i)
                break

    positions = [(i + 1) for i in positions]
    rec_positions = [1 / i for i in positions]
    result = {}
    result['mrr'] = np.mean(rec_positions)
    for value in hits:
        hits_at_value = [1 if pos <= value else 0 for pos in positions]
        result['hits@' + str(value)] = np.sum(hits_at_value) / len(hits_at_value)
    return result

