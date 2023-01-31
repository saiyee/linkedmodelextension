import json
import logging

from config import VCSLAM
from modelextension.evaluation.general_evaluator import GeneralEvaluator
from preprocess.steiner_tree import get_target_nodes, compute_steiner_tree
from util import parse
from visualization.visualize import visualize

geouris = ['http://tmdtkg#geoposition', 'http://tmdtkg#location', 'http://tmdtkg#latitude',
           'http://tmdtkg#longitude', 'http://tmdtkg#wgs84', 'http://tmdtkg#address',
           'http://tmdtkg#street', 'http://tmdtkg#geo_type', 'http://tmdtkg#district',
           'http://tmdtkg#utm','http://tmdtkg#northing','http://tmdtkg#easting']


class ModelRecommendationEvaluationResult:
    id:int
    percentages: list[float]
    target_nodes: set[str]
    recommended_nodes: set[str]

    def __init__(self, id):
        self.id = id
        self.percentages = []
        self.target_nodes = set()
        self.recommended_nodes = set()

    @property
    def recommended_geo_nodes(self) -> [str]:
        return [uri for uri in self.recommended_nodes if uri in geouris]

    @property
    def missed_nodes(self) -> [str]:
        return [uri for uri in self.target_nodes if uri not in self.recommended_nodes]

    @property
    def missed_geo_nodes(self) -> [str]:
        return [uri for uri in self.target_nodes if uri not in self.recommended_nodes and uri in geouris]

    @property
    def percentage_geo(self):
        if not self.recommended_nodes:
            return 0
        return len(self.recommended_geo_nodes) / len(self.recommended_nodes)

    @property
    def target_geo_uris(self):
        return [uri for uri in self.target_nodes if uri in geouris]

    @property
    def percentage_geo_recommended(self):
        if not self.target_geo_uris:
            return -1
        return len(self.recommended_geo_nodes) / len(self.target_geo_uris)

    @property
    def count(self):
        return len(self.recommended_nodes)

    @property
    def target_geo_nodes(self):
        return [uri for uri in self.target_nodes if uri in geouris]

    @property
    def average(self):
        if not self.percentages:
            return 0
        return sum(self.percentages) / len(self.percentages)

    def __repr__(self):
        return f"Result ({self.id}): {len(self.recommended_nodes)}/{len(self.target_nodes)} {self.average} {self.percentages}"


def evaluate_general_focus():
    model = VCSLAM()
    with open(model.PARSED_MODELS_PATH, 'r') as file:
        models = json.load(file)

    results = []
    start = 1
    length = 101

    for id, model in enumerate(models[start-1:start-1+length], start=start):
        result = ModelRecommendationEvaluationResult(id)
        recommended_nodes_set = set()
        # visualize(model)
        target_nodes = get_target_nodes(index=id-1)
        steiner_tree = compute_steiner_tree(index=id-1)
        # visualize(steiner_tree)
        if not target_nodes:
            # results.append(result)
            logging.info(f"Skipped model {id} as 0 target nodes are found")
            #visualize(model)
            #visualize(steiner_tree)
            continue
        result.target_nodes = target_nodes
        logging.info(f"Target ({id}) {len(target_nodes)} {target_nodes}")

        for count in range(1):
            evaluator = GeneralEvaluator(id, target_nodes, steiner_tree)
            recommended_nodes = evaluator.evaluate()
            recommended_nodes_set |= set(recommended_nodes)
            model_percentage = len(recommended_nodes) / len(target_nodes)
            result.percentages.append(model_percentage)

        result.recommended_nodes = recommended_nodes_set
        # logging.info(result)
        results.append(result)

    missing_geo = [x.missed_geo_nodes for x in results]
    return results


if __name__ == '__main__':
    evaluate_general_focus()
