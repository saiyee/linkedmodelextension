import json
import random
import time
from math import floor
from pathlib import Path

import numpy as np
import torch
import dgl

import config
from config import Environment as env, GNN as GNN_CONFIG
from linkprediction import utils
from linkprediction.rgcn import LinkPredict, node_norm_to_edge_norm
from util.parse import generate_dictionaries, generate_id_dict, encode_triples
import preprocess.modify_relations as modrel


def train_models(
        train_triples: [(str, str, str)],
        test_triples: [(str, str, str)],
        config: GNN_CONFIG,
        dataset: config.Dataset = env.DEFAULT_DATASET,
        classes_mapping=None,
        predicates_mapping=None):
    Path(dataset.TORCH_LP_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    classes, predicates = generate_dictionaries(train_triples)
    if not classes_mapping:
        classes_mapping = generate_id_dict(classes)

    if not predicates_mapping:
        predicates_mapping = generate_id_dict(predicates)

    print("training triples size", len(train_triples))

    train_data = encode_triples(train_triples, classes_mapping, predicates_mapping)
    valid_data = encode_triples(test_triples, classes_mapping, predicates_mapping)

    if env.LOG_LEVEL >= 10:
        print("Train data size: ", train_data.shape)
        print("Valid data size: ", valid_data.shape)

    # load graph data
    num_nodes = len(classes)
    num_rels = len(predicates)

    # check cuda
    use_cuda = config.GPU >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(config.GPU)

    # create model
    lp_model = LinkPredict(num_nodes,
                           config.N_HIDDEN,
                           num_rels,
                           num_bases=config.N_BASES,
                           num_hidden_layers=config.N_LAYERS,
                           dropout=config.DROPOUT,
                           use_cuda=use_cuda,
                           reg_param=config.REGULARIZATION)

    # save model
    torch.save(lp_model, dataset.TORCH_LP_MODEL_PATH)
    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(valid_data)

    # build test graph
    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)
    test_deg = test_graph.in_degrees(
        range(test_graph.number_of_nodes())).float().view(-1, 1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    if use_cuda:
        lp_model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(lp_model.parameters(), lr=config.LEARNING_RATE)

    forward_time = []
    backward_time = []

    # training loop
    # print("start training...")

    epoch = 0
    best_mrr = 100000
    best_hits3 = 0
    while True:
        lp_model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                train_data, config.GRAPH_BATCH_SIZE, config.GRAPH_SPLIT_SIZE,
                num_rels, adj_list, degrees, config.NEGATIVE_SAMPLE,
                config.EDGE_SAMPLER)
        # print("Done edge sampling")

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            data, labels = data.cuda(), labels.cuda()
            g = g.to(config.GPU)

        embed = lp_model(g, node_id, edge_type, edge_norm)
        loss = lp_model.get_loss(g, embed, data, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lp_model.parameters(), config.GRAD_NORM)  # clip gradients
        optimizer.step()
        optimizer.zero_grad()

        # validation
        if epoch % config.EVALUATE_EVERY_N == 0:
            lp_model.eval()
            # print("start eval")
            embed = lp_model(test_graph, test_node_id, test_rel, test_norm)
            results = utils.calc_mrr(embed, lp_model.w_relation, torch.LongTensor(train_data),
                                     valid_data, test_data, hits=[1, 3], eval_bz=config.EVAL_BATCH_SIZE,
                                     eval_p=config.MRR_EVAL_PROTOCOL)
            mrr = results['mrr']
            hits3 = results['hits@3']
            hits1 = results['hits@1']
            print(f"epoch {epoch} MRR {mrr:.2f} hits@1 {hits1:.2f} hits@3 {hits3:.2f}", end="")
            if best_hits3 < hits3:
                best_hits3 = hits3
            if best_mrr >= mrr:
                best_mrr = mrr
                torch.save({'state_dict': lp_model.state_dict(), 'epoch': epoch},
                           dataset.TORCH_LP_MODEL_STATE_PATH, _use_new_zipfile_serialization=False)
                print(f"***", end="")

            # if hits3 == 1:
            #    break

            if epoch >= config.N_EPOCHS:
                break

            if use_cuda:
                lp_model.cuda()

    # use best model checkpoint
    checkpoint = torch.load(dataset.TORCH_LP_MODEL_STATE_PATH)
    lp_model.eval()
    lp_model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    embed = lp_model(test_graph, test_node_id, test_rel, test_norm)
    torch.save(embed, dataset.TORCH_LP_EMBEDDING_PATH)
    results = utils.calc_mrr(embed, lp_model.w_relation, torch.LongTensor(train_data), valid_data,
                             test_data, hits=[1, 3], eval_bz=config.EVAL_BATCH_SIZE, eval_p=config.MRR_EVAL_PROTOCOL)
    mrr = results['mrr']
    hits3 = results['hits@3']
    hits1 = results['hits@1']
    print(f"Best epoch {epoch} MRR {mrr} hits@1 {hits1} hits@3 {hits3}")

    return results


def dgl_graphify(rdf_graph_list, node_dict, pred_dict):
    """Converts dictionary mapping graph identifier to corresponding dgl graph.

    Arguments:
      rdf_graph_list: List of rdf graphs to convert to dgl graphs
      pred_dict: Dictionary mapping predicates to their global ids.
      node_dict: Dictionary mapping nodes to their global ids.
    """
    graph_dict = {}

    for graph in rdf_graph_list:
        pred_global_ids = []
        src_global_ids = []
        dst_global_ids = []

        # record global numeric ids of the
        # subj, pred and obj of the dfl graph
        for subj, pred, obj in graph:
            pred_global_ids.append(pred_dict[pred])
            src_global_ids.append(node_dict[subj])
            dst_global_ids.append(node_dict[obj])

        # map global ids of nodes to local ids
        # to avoid errors that arise in cases like in
        # def test_less_nodes_than_it_seems() under
        # test_sanity.py
        local_dict = generate_id_dict(src_global_ids + dst_global_ids)
        src_local_ids = [local_dict[id] for id in src_global_ids]
        dst_local_ids = [local_dict[id] for id in dst_global_ids]

        # create dgl graph
        dgl_g = dgl.graph((src_local_ids, dst_local_ids))

        inv_local_dict = {v: k for k, v in local_dict.items()}

        node_feats = [inv_local_dict[node.tolist()] for node in dgl_g.gold_nodes()]

        # record global_ids through dgl_g's features
        dgl_g.edata["edge_type"] = torch.tensor(pred_global_ids)
        dgl_g.ndata["node_type"] = torch.tensor(node_feats)

        # have the graph be bidirectional. copy edata
        # dgl_g = dgl.add_reverse_edges(dgl_g, copy_edata=True)

        graph_dict[graph.identifier] = dgl_g

    return graph_dict


def get_train_and_val(rdf_graph_list, validation_graphs):
    """Extracts training and validation dataset according to validation graphs."""
    train_graph_list = []
    val_graph_list = []

    for graph in rdf_graph_list:

        val = False

        graph_id = str(graph.identifier)
        for val_graph in validation_graphs:
            if val_graph in graph_id:
                val = True
                break

        if val:
            val_graph_list.append(graph)
        else:
            train_graph_list.append(graph)

    return train_graph_list, val_graph_list


if __name__ == '__main__':
    model = env.DEFAULT_DATASET
    with open(model.PARSED_MODELS_PATH, 'r') as file:
        models = json.load(file)

    print('# of models', len(models))

    triples = []
    for model in models:
        for triple in model:
            triples.append(tuple(triple))

    print("models triples size", len(triples))

    # train GNN
    lp_config = GNN_CONFIG()
    test_triples = random.sample(triples, floor(len(triples) / 10))
    train_models(train_triples=triples, test_triples=test_triples, config=lp_config)
