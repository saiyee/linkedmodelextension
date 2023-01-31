import math
import os
import random
import json

import rdflib
from rdflib import Graph

import config
from config import Environment as env

ontology: [(str, str, str)] = []


def get_prediction_graph(prediction_triplets):
    val_triplets = [item for set in prediction_triplets for item in map_to_rdf(set)]

    prediction_graph = rdflib.Graph(identifier="valid")
    for triple in val_triplets:
        prediction_graph.add(triple)


def get_ontology(model: config.Dataset = env.DEFAULT_DATASET) -> [(str, str, str)]:
    if len(ontology) == 0:
        load_ontology(model.ONTOLOGY_PATH)
    return ontology


def load_models(models_path):
    models = []
    files = os.listdir(models_path)
    for file in files:
        rdf_graph = rdflib.Graph()
        rdf_graph.parse(models_path + file, format='turtle')
        model = []
        for (s, a, t) in rdf_graph.triples((None, None, None)):
            model.append((str(s).split('#')[1],
                          str(t).split('#')[1],
                          str(a).split('#')[1]))
        models.append(model)

    return models


def map_to_rdf(triples: [str], prefix: str = env.RDF_PREFIX) -> [
    (rdflib.term.URIRef, rdflib.term.URIRef, rdflib.term.URIRef)]:
    return [(rdflib.term.URIRef(prefix + tup[0]), rdflib.term.URIRef(prefix + tup[1]),
             rdflib.term.URIRef(prefix + tup[2])) for tup in triples]


def modify_dataset(models, prediction_triplets: [[]], weights: []):
    result = []

    filter_triplets = [map_to_rdf(set) for set in prediction_triplets]
    all_filter_triplets = [item for set in filter_triplets for item in set]

    selected_triplets = []
    for j, weight in enumerate(weights):
        for k in range(math.ceil(weight * len(models))):
            selected_triplets.append(filter_triplets[j])

    for i, model in enumerate(models):
        triples = [triple for triple in model.triples((None, None, None))]
        model_triplets = [t for t in triples if t not in all_filter_triplets]

        new_graph = rdflib.Graph()
        for triple in model_triplets:
            new_graph.add(triple)
        for triple in selected_triplets[i]:
            new_graph.add(triple)

        result.append(new_graph)

    return result


def load_ontology(ontology_path):
    if env.model == 'vcslam':
        rdf_graph = rdflib.Graph()
        rdf_graph.parse(ontology_path, format='turtle')
        for (s, a, t) in rdf_graph.triples((None, None, None)):
            (s, a, t) = (str(s), str(a), str(t))
            if 'label' not in a and 'domain' not in a and 'range' not in a and 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type' not in a:
                ontology.append((s, a, t))
    elif env.model == 'taheriyan':
        with open(ontology_path, 'r') as file:
            temp = json.load(file)
        file.close()
        for triple in temp:
            ontology.append(tuple(triple))

    return ontology


def generate_sampled_graphs(graph: rdflib.Graph, no_duplicates: int, no_missing_edges: int) -> [rdflib.Graph]:
    graphs = []

    triples = [triple for triple in graph.triples((None, None, None))]
    size = len(triples)

    for i in range(no_duplicates):

        sample = random.sample(triples, size - no_missing_edges)
        new_graph = rdflib.Graph()
        for triple in sample:
            new_graph.add(triple)

        graphs.append(new_graph)

    return graphs


def generate_noised_graphs(input_graphs: [rdflib.Graph], no_noised: int, no_missing_edges: int, val_triplets,
                           static_noise_facts_list: [[(str, str, str)]]) -> [rdflib.Graph]:
    noise_graphs = random.sample(input_graphs, no_noised)
    noise_models = []

    for i, noise_graph in enumerate(noise_graphs):
        triples = [triple for triple in noise_graph.triples((None, None, None))]

        static_noise_facts = static_noise_facts_list[i % 2]

        static_noise_triplets = map_to_rdf(static_noise_facts)
        noise_triplets = [t for t in triples if t not in val_triplets]
        new_graph = rdflib.Graph()
        for triple in noise_triplets:
            new_graph.add(triple)
        for triple in static_noise_triplets:
            new_graph.add(triple)

        noise_models.append(new_graph)

    return noise_graphs


def multiply_dataset(input_graph: Graph, factor: int) -> [Graph]:
    result = []
    for i in range(factor):
        result.append(input_graph)

    return result
