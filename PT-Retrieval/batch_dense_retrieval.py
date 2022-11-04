import argparse
import os
import csv
import glob
import json
import os
import pathlib

import argparse
import csv
import logging
import pickle
from typing import List, Tuple
import codecs
from tqdm import tqdm

import numpy as np
import torch
from torch import nn

from dpr.models import init_encoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params, add_tuning_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint,move_to_device
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer
from generate_dense_embeddings import gen_ctx_vectors
from dense_retriever import DenseRetriever, parse_qa_csv_file, load_passages, validate

csv.field_size_limit(10000000)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

def prepare_encoders(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    
    tensorizer, encoder, _ = init_encoder_components("dpr", args, inference_only=True)

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16,
                                            args.fp16_opt_level)
    encoder.eval()

    model_to_load = get_model_obj(encoder)
    model_to_load.load_state_dict(saved_state.model_dict, strict=False)

    ctx_model = model_to_load.ctx_model
    question_model = model_to_load.question_model

    return ctx_model, question_model, tensorizer

def batch_retrieval(args, topic, ctx_model, retriever, tensorizer):
    logger.info("------------------------------------------------")
    logger.info(f"Evaluate {topic}")
    args.ctx_file = os.path.join(args.ctx_dir, f"{topic}-papers-10k.tsv")
    args.out_file = os.path.join(args.out_dir, f"encoded-{topic}")

    logger.info(f"- Generating {args.ctx_file}")
    logger.info(f"- To {args.out_file}")
    rows = []
    with open(args.ctx_file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        # file format: doc_id, doc_text, title
        rows.extend([(row[0], row[1], row[2]) for row in reader if row[0] != 'id'])

    data = gen_ctx_vectors(rows, ctx_model, tensorizer, args, True) 

    file = args.out_file + '.pkl'
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info('Writing results to %s' % file)
    with open(file, mode='wb') as f:
        pickle.dump(data, f)

    logger.info('Total passages processed %d. Written to %s', len(data), file)
    # ===============================

    args.qa_file = os.path.join(args.ctx_dir, f"{topic}-questions.tsv")
    args.encoded_ctx_file = os.path.join(args.out_dir, f"encoded-{topic}.pkl")
    input_paths = glob.glob(args.encoded_ctx_file)
    logger.info('Reading all passages data from files: %s', input_paths)
    retriever.index.index_data(input_paths)

    questions = []
    question_answers = []

    for ds_item in parse_qa_csv_file(args.qa_file):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)
    questions_tensor = retriever.generate_question_vectors(questions)
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)
    all_passages = load_passages(args.ctx_file)

    check_index = 1 if args.oagqa else 0
    questions_doc_hits, top_k_hits = validate(all_passages, question_answers, top_ids_and_scores, args.validation_workers,
                                  args.match, check_index)

    return top_k_hits



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)
    add_tuning_params(parser)

    parser.add_argument('--ctx_dir', required=True, type=str, default=None, help='Path to passages set .tsv file')
    parser.add_argument('--out_dir', required=True, type=str, default=None, help='output file path to store encoded passages to')
    parser.add_argument('--input_file', required=True, type=str, default=None, help='input file path to evalute')
    parser.add_argument('--result_file', required=True, type=str, default=None, help='output file path to write results to')
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument('--index_buffer', type=int, default=50000, help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument('--validation_workers', type=int, default=16, help="Number of parallel processes to validate results")
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'], help="Answer matching logic type")
    parser.add_argument('--n-docs', type=int, default=20, help="Amount of top docs to return")
    parser.add_argument("--oagqa", action='store_true')
    args = parser.parse_args()
    setup_args_gpu(args)

    ctx_model, question_model, tensorizer = prepare_encoders(args)    
    vector_size = question_model.get_out_size()
    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size, args.index_buffer)
    else:
        index = DenseFlatIndexer(vector_size, args.index_buffer)

    retriever = DenseRetriever(question_model, args.batch_size, tensorizer, index)

    eval_list = []
    with open(args.input_file, 'r') as f:
        for line in f.readlines():
            eval_list.append(line.strip())

    print("Eval list =", eval_list)

    result_dict = {}

    for topic in eval_list:
        top_k_hits = batch_retrieval(args, topic, ctx_model, retriever, tensorizer)
        result_dict[topic] = top_k_hits[-1]

    scores = 0
    for topic, value in result_dict.items():
        scores += value
    avg_score = scores / len(result_dict)
    result_dict["average"] = avg_score
    
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    json.dump(result_dict, open(args.result_file, 'w'))