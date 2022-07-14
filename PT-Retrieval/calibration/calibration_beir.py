import os
import sys
import logging
import argparse
from pathlib import Path

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

BASE_DIR = Path.resolve(Path(__file__)).parent.parent
sys.path.append(str(BASE_DIR))

from beir_eval.models import DPRForBeir
from calibration.utils import calculate_ece, calculate_erce, plot_calibration_curve
from dpr.options import add_encoder_params, setup_args_gpu, set_seed, add_tokenizer_params, add_cuda_params, add_tuning_params

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def main(args):
    model = DRES(DPRForBeir(args), batch_size=256)

    dataset = args.dataset

    data_path = "beir_eval/datasets/%s" % dataset

    logging.info("evaluating dataset %s..." % dataset)

    if not os.path.exists(data_path):
    #### Download NFCorpus dataset and unzip the dataset
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = "beir_eval/datasets"
        util.download_and_unzip(url, out_dir)
        data_path = os.path.join(out_dir, dataset)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=args.split)

    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    ece = calculate_ece(qrels, results, args)
    print("ECE = %.3f" % ece)

    ece = calculate_erce(qrels, results, args)
    print("ERCE = %.3f" % ece)

    # nq2, webq, trivia, curatedtrec
    plot_calibration_curve(qrels, results, args)


if __name__ == '__main__':

#### /print debug information to stdout

    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)
    add_tuning_params(parser)

    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    parser.add_argument('--dataset', type=str) 
    parser.add_argument('--split', type=str, default="text") 
    parser.add_argument('--n_non_rel', type=int, default=5) 
    parser.add_argument('--n_ece_docs', type=int, default=5) 
    parser.add_argument('--n_erce_docs', type=int, default=20) 


    args = parser.parse_args()
    args.dpr_mode = False
    args.fix_ctx_encoder = False
    args.use_projection = False

    setup_args_gpu(args)

    main(args)

