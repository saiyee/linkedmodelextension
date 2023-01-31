class EvaluationConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    identifier: str
    default: False
    positives: [str]
    target: str
    _target_nodes: [str]
    _prediction_triplets: [(str, str, str)]

    @property
    def target_nodes(self):
        if not self._target_nodes:
            for triple in self.prediction_triplets:
                self._target_nodes.append(triple[2])
            self._target_nodes = list(set(self._target_nodes))
        return self._target_nodes

    @property
    def prediction_triplets(self):
        return [(l[0], l[1], l[2]) for l in self._prediction_triplets]


def intersection(list1, list2):
    intersec = [value for value in list1 if value in list2]
    return intersec
