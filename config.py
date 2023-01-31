import logging

class Dataset:
    def __init__(self, identifier: str, base_path: str):
        self.identifier = identifier
        self.BASE_PATH = base_path
        self.EVALUATION_CONFIG_PATH = self.BASE_PATH + "/evaluation_config.json"

        self.ONTOLOGY_PATH = self.BASE_PATH + '/ontology/ontology.ttl'
        self.MODELS_PATH = self.BASE_PATH + '/rdf'
        self.LABEL_MAPPINGS_PATH = self.BASE_PATH + '/label_mappings'
        self.PARSED_MODELS_PATH = self.BASE_PATH + '/' + identifier + '.models.json'

        self.CLASSES_MAPPING_PATH = self.BASE_PATH + '/mappings/classes_mapping.json'
        self.PREDICATES_MAPPING_PATH = self.BASE_PATH + '/mappings/predicates_mapping.json'

        self.WORD2VEC_MODEL_PATH = self.BASE_PATH + '/models/word2vec.emb'
        self.PECAN_MODEL_PATH = self.BASE_PATH + '/models/pecan.emb'
        self.SR_MODEL_PATH = self.BASE_PATH + '/models/statistics_recommender.json'

        self.TORCH_LP_MODEL_PATH = self.BASE_PATH + '/models/lp_model.pt'
        self.TORCH_LP_EMBEDDING_PATH = self.BASE_PATH + '/models/embeddings.pt'
        self.TORCH_LP_MODEL_STATE_PATH = self.BASE_PATH + '/models/model_state.pth'


class ARMSTRADER(Dataset):
    def __init__(self):
        Dataset.__init__(self, 'armstrader', 'data/armstrader')

class FB15K(Dataset):
    def __init__(self):
        Dataset.__init__(self, 'fb15k', 'data/fb15k')

class VCSLAM(Dataset):
    def __init__(self):
        Dataset.__init__(self, 'vcslam', 'data/vc-slam')

class MUSEUM(Dataset):
    def __init__(self):
        Dataset.__init__(self, 'museum', 'data/museum')


class Environment:
    RDF_PREFIX = "http://tmdtkg#"

    # 5: TRACE -> All info
    # 10: DEBUG -> Some info
    # 20: INFO -> Basic info
    LOG_LEVEL = 20

    # 0 for working with ontology based node2vec, 1 for working with model based node2vec
    NODE2VEC_SELECT = 1

    model = 'armstrader'

    if model == 'vcslam':
        DEFAULT_DATASET = VCSLAM()

    if model == 'armstrader':
        DEFAULT_DATASET = ARMSTRADER()

    if model == 'museum':
        DEFAULT_DATASET = MUSEUM()

class NodeEmbeddings:
    def __init__(self,
                 p=1,
                 q=1,
                 workers=4,
                 walk_length=30,
                 walks=1000,
                 window=10,
                 dimension=100,
                 negative=5,
                 ns_exponent=1.0):
        self.p = p
        self.q = q
        self.workers = workers
        self.walk_length = walk_length
        self.walks = walks
        self.window = window
        self.dimension = dimension
        self.negative = negative
        self.ns_exponent = ns_exponent


class GNN:
    # dropout probability
    DROPOUT = 0.1
    # number of hidden units
    N_HIDDEN = 100
    # number of propagation rounds
    N_LAYERS = 2
    # gpu
    GPU = -1
    # learning rate
    LEARNING_RATE = 1e-3
    # number of weight blocks for each relation
    N_BASES = 10
    # number of minimum training epochs
    N_EPOCHS = 5000
    # batch size when evaluating
    EVAL_BATCH_SIZE = 100
    # type of evaluation protocol: 'raw' or 'filtered' mrr
    MRR_EVAL_PROTOCOL = "filtered"
    # regularization weight
    REGULARIZATION = 0.01
    # norm to clip gradient to
    GRAD_NORM = 1.0
    # number of edges to sample in each iteration
    GRAPH_BATCH_SIZE = 20
    # portion of edges used as positive sample
    GRAPH_SPLIT_SIZE = 0.3
    # number of negative samples per positive sample
    NEGATIVE_SAMPLE = 5
    # perform evaluation every n epochs
    EVALUATE_EVERY_N = 100
    # type of edge sampler: 'uniform' or 'neighbor'
    EDGE_SAMPLER = "neighbor"


logging.addLevelName(5,"TRACE")
logging.basicConfig(level=Environment.LOG_LEVEL, format='%(asctime)s [%(levelname)s] - %(message)s')
