import json
import random
from pathlib import Path
from typing import Dict

from rdflib import RDF, URIRef, Literal, Graph, XSD

from config import ARMSTRADER
from config import Environment as env
from evaluate.EvaluationConfig import EvaluationConfig
from util.parse import get_graphs_from_rdf


def parse_armstrader_data():
    selected_model = ARMSTRADER()

    graphs = get_graphs_from_rdf(selected_model.BASE_PATH + "/raw")
    type_mapping = {}
    models = []
    Path(selected_model.MODELS_PATH).mkdir(exist_ok=True)
    counter =0


    for graph in graphs:
        for (sub, _, obj) in graph.triples((None, RDF.type, None)):
            type_mapping[sub] = obj

    for graph in graphs:
        model = []
        for (sub, pred, obj) in graph.triples((None, None, None)):
            counter += 1

            if pred == RDF.type:
                continue

            if sub in type_mapping:
                sub = type_mapping[sub]
            else:
                continue

            if obj in type_mapping:
                obj = type_mapping[obj]
            elif isinstance(obj, Literal):
                obj = XSD.string

            model.append((str(sub), str(pred), str(obj)))
        models.append(model)

    print("total triples", counter)
    with open(selected_model.PARSED_MODELS_PATH, 'w') as file:
        json.dump(models, file, indent=4)
        file.close()


if __name__ == '__main__':
    parse_armstrader_data()
