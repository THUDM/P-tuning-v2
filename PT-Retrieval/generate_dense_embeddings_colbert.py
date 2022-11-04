#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
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

import sys
sys.path.append("./colbert")

from colbert.utils.parser import Arguments
from colbert.evaluation.loaders import load_colbert
# from colbert.parameters import DEVICE
# from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.modeling.inference import ModelInference
# from colbert.modeling.maxsim import Maxsim


logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

csv.field_size_limit(10000000)


def gen_ctx_vectors(ctx_rows: List[Tuple[object, str, str]], inference,
                    args, insert_title: bool = True) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = args.batch_size
    total = 0
    results = []
    for j, batch_start in tqdm(enumerate(range(0, n, bsz)), total=round(n/bsz)):
        batch = [ctx[2] + '. ' + ctx[1] for ctx in ctx_rows[batch_start:batch_start + bsz]]
        embs = inference.docFromText(batch, bsize=32, keep_dims=True)
        print(embs.shape)
        assert len(embs) == len(batch)
        out = embs
        out = out.cpu()

        ctx_ids = [r[0] for r in ctx_rows[batch_start:batch_start + bsz]]

        assert len(ctx_ids) == out.size(0)

        total += len(ctx_ids)
        results.append(out.numpy())

    results = np.concatenate(results, axis=0)
    print(results.shape)
    return results


def main(args):
    colbert, _ = load_colbert(args)
    colbert.cuda()
    colbert.eval()
    inference = ModelInference(colbert, amp=args.amp)
    # query_tokenizer = QueryTokenizer(args.query_maxlen)
    # doc_tokenizer = DocTokenizer(args.doc_maxlen)

    logger.info('reading data from file=%s', args.ctx_file)

    rows = []
    with open(args.ctx_file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        # file format: doc_id, doc_text, title
        rows.extend([(row[0], row[1], row[2]) for row in reader if row[0] != 'id'])

    shard_size = int(len(rows) / args.num_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info('Producing encodings for passages range: %d to %d (out of total %d)', start_idx, end_idx, len(rows))
    rows = rows[start_idx:end_idx]

    data = gen_ctx_vectors(rows, inference, args, True) 

    file = args.out_file + '-' + str(args.shard_id) + '.pkl'
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info('Writing results to %s' % file)
    with open(file, mode='wb') as f:
        pickle.dump(data, f)

    logger.info('Total passages processed %d. Written to %s', len(data), file)


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
    parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file')
    parser.add_argument('--out_file', required=True, type=str, default=None,
                        help='output file path to write results to')
    parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
    parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")

    parser.add_model_parameters()
    parser.add_model_training_parameters()

    args = parser.parse()
    args.local_rank = args.rank
    setup_args_gpu(args)

    
    main(args)