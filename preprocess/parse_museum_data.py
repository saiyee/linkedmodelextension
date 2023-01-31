import json
from pathlib import Path

from rdflib import RDF, URIRef, Literal, Graph, XSD

from config import MUSEUM
from util.parse import get_graphs_from_rdf


def parse_museum_data():
    selected_model = MUSEUM()
    graphs = get_graphs_from_rdf(selected_model.BASE_PATH + "/raw")
    type_mapping = {}
    models = []
    Path(selected_model.MODELS_PATH).mkdir(exist_ok=True)

    for graph in graphs:
        for (sub, _, obj) in graph.triples((None, RDF.type, None)):
            type_mapping[sub] = obj

    for graph in graphs:
        mapped_graph = Graph()
        model = []
        for (sub, pred, obj) in graph.triples((None, None, None)):
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
            else:
                continue

            mapped_graph.add((sub, pred, obj))
            model.append((str(sub), str(pred), str(obj)))
        models.append(model)

        # mapped_graph.serialize(destination=selected_model.MODELS_PATH + "/"
        #                                   + graph.identifier + ".ttl")

    with open(selected_model.PARSED_MODELS_PATH, 'w') as file:
        json.dump(models, file, indent=4)
        file.close()

    triples = []
    for model in models:
        for triple in model:
            triples.append(tuple(triple))

    with open(selected_model.ONTOLOGY_PATH, 'w') as file:
        json.dump(triples, file, indent=4)
    file.close()


if __name__ == '__main__':
    parse_museum_data()
