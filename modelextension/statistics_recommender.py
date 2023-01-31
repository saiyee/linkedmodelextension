import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import config
from config import Environment as env

import numpy as np


class StatisticsRecommender:
    triples: [(str, str, str)]
    matrix: np.array

    def __init__(self, triples: [(int, int, int)]):
        self.triples = triples
        self.matrix = np.array(triples, dtype='float64')

    def most_similar_classes(self, anchors: [str], topn: int = 5) -> [(str, float)]:
        # for each of the positives, generate a list of most likely values
        neighbors = [self.__get_neighbors(term) for term in anchors]
        # merge the lists
        neighbors = [x for xs in neighbors for x in xs]
        # count terms
        counter = Counter(neighbors)
        total = sum(list(counter.values())[:topn])
        matches = sorted(list(counter.items()), key=lambda x: x[1], reverse=True)
        matches = [(concept, count / total) for concept, count in matches]

        return matches[:topn]

    def __get_neighbors(self, anchor: str):
        return list(set([t for (s, a, t) in self.triples if s == anchor]))

    def predict_link(self,anchor: int, candidate: int):
        array = self.matrix
        max_id = np.max(array[:, 1]) + 1
        filtered = array[np.logical_and(array[:, 0] == anchor, array[:, 2] == candidate), :]
        predicates = filtered[:, 1]
        pred_count = np.unique(predicates, return_counts=True)
        pred_count = np.transpose(pred_count)
        pred_count[:, 1] = pred_count[:, 1] / len(filtered)
        pred_count = pred_count[pred_count[:, 1].argsort()]

        result = np.zeros([1, np.int(max_id)], dtype='float64')
        for i in range(len(pred_count)):
            result[0, int(pred_count[i][0])] = pred_count[i][1]
        return result

    def predict_links(self, tuples: [(int, int)]):
        result = None
        for tup in tuples:
            row = self.predict_link(anchor=tup[0], candidate=tup[1])
            if result is None:
                result = np.empty([0, row.shape[1]])
            result = np.vstack((result, row))
        if result is None:
            result = np.empty([0, 0])
        return result
