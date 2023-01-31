import json
import random
from pathlib import Path
from typing import Dict

from rdflib import RDF, URIRef, Literal, Graph, XSD

from config import ARMSTRADER, FB15K
from config import Environment as env
from evaluate.EvaluationConfig import EvaluationConfig
from util.parse import get_graphs_from_rdf


def parse_fb15k_data():
    dataset = FB15K()

    triples = []
    f = open(dataset.BASE_PATH + "/raw" + "/train.txt", "r")
    for x in f:
        token = x.split(sep="\t")
        triple = tuple(token)
        triples.append(triple)

    models = []
    Path(dataset.MODELS_PATH).mkdir(exist_ok=True)

    models.append(triples)

    with open(dataset.PARSED_MODELS_PATH, 'w') as file:
        json.dump(models, file, indent=4)
        file.close()


if __name__ == '__main__':
    parse_fb15k_data()
