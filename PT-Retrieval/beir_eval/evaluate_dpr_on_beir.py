from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from models import DPRForBeir

import sys
from pathlib import Path
BASE_DIR = Path.resolve(Path(__file__)).parent.parent
sys.path.append(str(BASE_DIR))
print(BASE_DIR)


from dpr.options import add_encoder_params, setup_args_gpu, \
    add_tokenizer_params, add_cuda_params, add_tuning_params
    
import argparse
import logging
import pathlib, os
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()

add_encoder_params(parser)
add_tokenizer_params(parser)
add_cuda_params(parser)
add_tuning_params(parser)

parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
parser.add_argument('--dataset', type=str) 
parser.add_argument('--split', type=str, default="text") 

args = parser.parse_args()
args.dpr_mode = False
args.fix_ctx_encoder = False
args.use_projection = False

setup_args_gpu(args)

model = DRES(DPRForBeir(args), batch_size=args.batch_size)

dataset = args.dataset

data_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets", dataset)

logging.info("evaluating dataset %s..." % dataset)
logging.info(data_path)
if not os.path.exists(data_path):
    #### Download NFCorpus dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    util.download_and_unzip(url, out_dir)
    data_path = os.path.join(out_dir, dataset)

#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(args.split)

retriever = EvaluateRetrieval(model, score_function="dot")

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
# logging.info("Query : %s\n" % queries[query_id])

# for rank in range(top_k):
#     doc_id = scores_sorted[rank][0]
#     # Format: Rank x: ID [Title] Body
#     logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
