import json


class EvaluationResult:

    def __init__(self, node: str,
                 positives: [str],
                 targets: [(str, str, str)],
                 result: [((str, str, str), float)],
                 identifier: str = None):
        self.identifier = identifier
        self.result = result
        self.targets = targets
        self.positives = positives
        self.node = node

    @property
    def matches(self):
        return [triple for triple in self.result if triple[0] in self.targets]

    @property
    def accuracy(self):
        return len(self.matches) / len(self.targets)

    def __str__(self) -> str:
        results = '\n'.join(['*' + str(result) + "(" + str(round(score.item(), 3)) + ")" if (result,score) in self.matches
                             else str(result) + "(" + str(round(score.item(), 3)) + ")" for result, score in
                             self.result])
        return f"Result | '{self.identifier}' (+{self.positives}) | {round(self.accuracy, 3)} ({len(self.matches)}/{len(self.targets)})\n{results}"
