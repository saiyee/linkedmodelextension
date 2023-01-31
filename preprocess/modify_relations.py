import logging
import random
from typing import Dict, List, Tuple


def get_relation_counts(triples: [(str, str, str)]):
    counts = {}

    for (s, a, t) in triples:
        if not (s, a, t) in counts:
            counts[(s, a, t)] = 0
        counts[(s, a, t)] += 1

    sorted_counts = sorted(counts.items(), reverse=True, key=lambda x: x[1])
    return sorted_counts


def reduce_relations(triples: List[Tuple[str, str, str]], targets: Dict[Tuple[str, str, str], float], verbose=False) -> List[
    Tuple[str, str, str]]:
    if verbose:
        counts = get_relation_counts(triples)
        counts = [[triple, count] for triple, count in counts if triple in targets]
        log_triples(counts, "PRE")

    modified_triples = []

    for triple in triples:
        if triple in targets:
            perc = targets[triple]
            if random.random() > perc:
                continue
        modified_triples.append(triple)

    if verbose:
        counts = get_relation_counts(modified_triples)
        counts = [[triple, count] for triple, count in counts if triple in targets]
        log_triples(counts, "POST")
    return modified_triples


def log_triples(triples, prefix="", level=logging.INFO):
    logging.log(level, prefix + " " + str(len(triples)) + " triples\n" + "\n".join([str(triple) for triple in triples]))
