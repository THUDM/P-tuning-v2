#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""
import argparse
import os
import csv
import glob
import logging
import time
from scipy.special import softmax

from utils import calibration_curve_with_ece

from dpr.models import init_encoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
     add_tokenizer_params, add_cuda_params, add_tuning_params
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer

from dense_retriever import DenseRetriever, parse_qa_csv_file, validate, load_passages, save_results, iterate_encoded_files

csv.field_size_limit(10000000)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def main(args):
    args.qa_file = f"data/retriever/qas/{args.dataset}-test.csv"
    if args.dataset == "curatedtrec":
        args.match = "regex"

    total_time = time.time()
    if not args.adapter:
        saved_state = load_states_from_checkpoint(args.model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_encoder_components("dpr", args, inference_only=True)

    encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')
    if args.adapter:
        adapter_name = model_to_load.load_adapter(args.model_file+".q")
        model_to_load.set_active_adapters(adapter_name)
    else:
        encoder_name = "question_model." 
        prefix_len = len(encoder_name)
        if args.prefix or args.prompt:
            encoder_name += "prefix_encoder."
        question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                                key.startswith(encoder_name)}
        model_to_load.load_state_dict(question_encoder_state, strict=not args.prefix and not args.prompt)
    vector_size = model_to_load.get_out_size()
    logger.info('Encoder vector_size=%d', vector_size)

    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size, args.index_buffer)
    else:
        index = DenseFlatIndexer(vector_size, args.index_buffer)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)


    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)

    index_path = "_".join(input_paths[0].split("_")[:-1])
    if args.save_or_load_index and (os.path.exists(index_path) or os.path.exists(index_path + ".index.dpr")):
        retriever.index.deserialize_from(index_path)
    else:
        logger.info('Reading all passages data from files: %s', input_paths)
        retriever.index.index_data(input_paths)
        if args.save_or_load_index:
            retriever.index.serialize(index_path)
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
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)

    logger.info('sample time: %f sec.', time.time() - sample_time)
    all_passages = load_passages(args.ctx_file)

    logger.info('total time: %f sec.', time.time() - total_time)
    if len(all_passages) == 0:
        raise RuntimeError('No passages data found. Please specify ctx_file param properly.')

    check_index = 1 if args.oagqa else 0
    questions_doc_hits, _ = validate(all_passages, question_answers, top_ids_and_scores, args.validation_workers,
                                  args.match, check_index)

    ece = calculate_ece(top_ids_and_scores, questions_doc_hits)
    print("ECE = %.3f" % ece)


def calculate_ece(top_ids_and_scores, questions_doc_hits):
    y_true = []
    y_prob = []

    for ids_and_scores, doc_hits in zip(top_ids_and_scores, questions_doc_hits):
        _, scores = ids_and_scores
        scores = scores[:args.n_docs]
        doc_hits = doc_hits[:args.n_docs]
        scores = softmax(scores)
        for pred_score, hit in zip(scores, doc_hits):
            y_true.append(int(hit))
            y_prob.append(pred_score)

    _, _, ece = calibration_curve_with_ece(y_true, y_prob, n_bins=10)
    return ece

# def calculate_erce(top_ids_and_scores, questions_doc_hits):
#     y_true = []
#     y_prob = []

#     for ids_and_scores, doc_hits in zip(top_ids_and_scores, questions_doc_hits):
#         _, scores = ids_and_scores
#         pos_scores = []
#         neg_scores = []

#         for pred_score, hit in zip(scores[:args.n_docs], doc_hits[:args.n_docs]):
#             if hit:
#                 pos_scores.append(pred_score)
#             else:
#                 neg_scores.append(pred_score)
#         if len(pos_scores) == 0:
#             for pred_score, hit in zip(scores, doc_hits):
#                 if hit:
#                     pos_scores.append(pred_score)
#                     break
     
#         for pscore in pos_scores:
#             for nscore in neg_scores:
#                 pos_score, neg_score = softmax([pscore, nscore])
#                 diff = pos_score - neg_score
#                 y_prob.append(abs(diff))
#                 y_true.append(int(diff > 0))

#     _, _, erce = calibration_curve(y_true, y_prob, n_bins=10)
#     return erce

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)
    add_tuning_params(parser)

    parser.add_argument('--dataset', type=str, choices=["nq", "trivia", "webq", "curatedtrec"]) 
    parser.add_argument('--ctx_file', required=True, type=str, default=None,
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    parser.add_argument('--encoded_ctx_file', type=str, default=None,
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
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

    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)
    print_args(args)
    main(args)
