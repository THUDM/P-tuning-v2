#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""
# import mkl
# mkl.get_max_threads()
import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator
from tqdm import tqdm

import numpy as np
import torch
from torch import Tensor as T

from dpr.data.qa_validation import calculate_matches
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params, add_tuning_params

import sys
sys.path.append("./colbert")

from colbert.utils.parser import Arguments
from colbert.evaluation.loaders import load_colbert
from colbert.parameters import DEVICE
from colbert.modeling.inference import ModelInference
from colbert.modeling.maxsim import Maxsim


csv.field_size_limit(10000000)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """
    def __init__(self, inference, batch_size: int, ctx_vectors, maxsim):
        self.inferece = inference
        self.batch_size = batch_size
        self.ctx_vectors = ctx_vectors
        self.maxsim = maxsim

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                queries = questions[batch_start:batch_start + bsz]
                out = self.inferece.queryFromText(queries, bsize=256)

                query_vectors.extend(out.split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded queries %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info('Total encoded queries tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs(self, query_vectors, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        results = []
        print(query_vectors.shape, self.ctx_vectors.shape)
        Q = query_vectors.cuda()
        D = self.ctx_vectors.cuda()
        print(Q.shape, D.shape)
        scores = self.maxsim(Q, D).cpu().numpy()
        for qid in tqdm(range(len(scores))):
            query_scores = scores[qid]
            # ranked_scores = sorted(query_scores.items(), key=lambda x:x[1],reverse=True)[:top_docs]
            sorted_index = np.argsort(query_scores)[::-1][:top_docs]
            # ids = [t[0] for t in ranked_scores]
            sorted_index = [int(i) for i in sorted_index]
            sorted_scores = [query_scores[i] for i in sorted_index]
            sorted_index = [str(i) for i in sorted_index]
            results.append((sorted_index, sorted_scores))
        # time0 = time.time()
        # results = self.index.search_knn(query_vectors, top_docs)
        # logger.info('index search time: %f sec.', time.time() - time0)
        return results


def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers


def validate(passages: Dict[object, Tuple[str, str]], answers: List[List[str]],
             result_ctx_ids: List[Tuple[List[object], List[float]]],
             workers_num: int, match_type: str, index: int = 0) -> List[List[bool]]:
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type, index)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    
    return match_stats.questions_doc_hits, top_k_hits


def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info('Reading data from: %s', ctx_file)
    if ctx_file.endswith(".gz"):
        with gzip.open(ctx_file, 'rt') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    return docs



def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info('Reading file %s', file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def main(args):
    total_time = time.time()
  
    # if args.hnsw_index:
    #     index = DenseHNSWFlatIndexer(vector_size, args.index_buffer)
    # else:
    #     index = DenseFlatIndexer(vector_size, args.index_buffer)
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)
    print("CTX:", input_paths)
    assert len(input_paths) == 1
    ctx_vectors = pickle.load(open(input_paths[0], 'rb'))
    ctx_vectors = torch.from_numpy(ctx_vectors).cuda()
    print(ctx_vectors.shape)

    colbert, _ = load_colbert(args)
    colbert.cuda()
    colbert.eval()
    inference = ModelInference(colbert, amp=args.amp)

    maxsim = Maxsim(args)
    maxsim.to(DEVICE)



    # index all passages
    

    retriever = DenseRetriever(inference, args.batch_size, ctx_vectors, maxsim)

    # index_path = "_".join(input_paths[0].split("_")[:-1])
    # if args.save_or_load_index and (os.path.exists(index_path) or os.path.exists(index_path + ".index.dpr")):
    #     retriever.index.deserialize_from(index_path)
    # else:
    #     logger.info('Reading all passages data from files: %s', input_paths)
    #     retriever.index.index_data(input_paths)
    #     if args.save_or_load_index:
    #         retriever.index.serialize(index_path)
    # get questions & answers
    questions = []
    question_answers = []

    for ds_item in parse_qa_csv_file(args.qa_file):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)

    sample_time = time.time()
    questions_tensor = retriever.generate_question_vectors(questions)

    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor, args.n_docs)

    logger.info('sample time: %f sec.', time.time() - sample_time)
    all_passages = load_passages(args.ctx_file)

    logger.info('total time: %f sec.', time.time() - total_time)
    if len(all_passages) == 0:
        raise RuntimeError('No passages data found. Please specify ctx_file param properly.')

    check_index = 1 if args.oagqa else 0
    questions_doc_hits, top_k_hits = validate(all_passages, question_answers, top_ids_and_scores, args.validation_workers,
                                  args.match, check_index)

    return top_k_hits

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    parser = Arguments(description='Training ColBERT with <query, positive passage, negative passage> triples.')

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    # add_cuda_params(parser)
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    add_tuning_params(parser)

    parser.add_argument('--encoder_type', type=str, choices=['dpr', 'moco', 'xmoco', 'simsiam'])
    parser.add_argument('--qa_file', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--ctx_file', required=True, type=str, default=None,
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    parser.add_argument('--encoded_ctx_file', type=str, default=None,
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--out_file', type=str, default=None,
                        help='output .json file path to write results to ')
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'],
                        help="Answer matching logic type")
    parser.add_argument('--n-docs', type=int, default=200, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')
    parser.add_argument("--oagqa", action='store_true')

    # args = parser.parse_args()

    # assert args.model_file, 'Please specify --model_file checkpoint to init model weights'
    parser.add_model_parameters()
    parser.add_model_training_parameters()
    # parser.add_training_input()
    # parser.add_beir_parameters()

    args = parser.parse()
    args.local_rank = args.rank
    setup_args_gpu(args)
    print_args(args)
    main(args)