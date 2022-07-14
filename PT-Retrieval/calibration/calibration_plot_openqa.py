import os
import sys
import logging
import argparse
from pathlib import Path

from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

BASE_DIR = Path.resolve(Path(__file__)).parent.parent
sys.path.append(str(BASE_DIR))

from beir_eval.models import DPRForBeir
from calibration.utils import plot_calibration_curve
from calibration.calibration_beir import plot_calibration_curve
from dpr.options import add_encoder_params, setup_args_gpu, set_seed, add_tokenizer_params, add_cuda_params, add_tuning_params
from convert_openqa_data import convert

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def main(args):
    model = DRES(DPRForBeir(args), batch_size=256)

    dataset = args.dataset

    data_path = "beir_eval/datasets/openqa-%s" % dataset

    logging.info("evaluating dataset %s..." % dataset)

    if not os.path.exists(data_path):
        logging.info("Convert OpenQA data to corpus, queries and qrels")
        convert(dataset)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    retriever = EvaluateRetrieval(model, score_function="dot")

    results = retriever.retrieve(corpus, queries)

    plot_calibration_curve(qrels, results, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)
    add_tuning_params(parser)

    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    parser.add_argument('--dataset', type=str, choices=["nq", "trivia", "webq", "curatedtrec"]) 
    parser.add_argument('--n_non_rel', type=int, default=5) 

    args = parser.parse_args()
    args.dpr_mode = False
    args.fix_ctx_encoder = False
    args.use_projection = False

    setup_args_gpu(args)
    # set_seed(args)

    main(args)

