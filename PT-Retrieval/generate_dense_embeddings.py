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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

csv.field_size_limit(10000000)
def gen_ctx_vectors(ctx_rows: List[Tuple[object, str, str]], model: nn.Module, tensorizer: Tensorizer,
                    args, insert_title: bool = True) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = args.batch_size
    total = 0
    results = []
    for j, batch_start in tqdm(enumerate(range(0, n, bsz)), total=round(n/bsz)):

        batch_token_tensors = [tensorizer.text_to_tensor(ctx[1], title=ctx[2] if insert_title else None) for ctx in
                               ctx_rows[batch_start:batch_start + bsz]]

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0),args.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch),args.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch),args.device)
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        ctx_ids = [r[0] for r in ctx_rows[batch_start:batch_start + bsz]]

        assert len(ctx_ids) == out.size(0)

        total += len(ctx_ids)

        results.extend([
            (ctx_ids[i], out[i].view(-1).numpy())
            for i in range(out.size(0))
        ])

    return results


def main(args):
    if not args.adapter:
        saved_state = load_states_from_checkpoint(args.model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)
    print_args(args)
    
    tensorizer, encoder, _ = init_encoder_components("dpr", args, inference_only=True)

    encoder = encoder.ctx_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16,
                                            args.fp16_opt_level)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')
    if args.adapter:
        adapter_name = model_to_load.load_adapter(args.model_file+".q")
        model_to_load.set_active_adapters(adapter_name)
    else:
        logger.debug('saved model keys =%s', saved_state.model_dict.keys())

        encoder_name = "ctx_model."
        prefix_len = len(encoder_name)
        if args.prefix or args.prompt:
            encoder_name += "prefix_encoder."

        ctx_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                    key.startswith(encoder_name)}
        model_to_load.load_state_dict(ctx_state, strict=not args.prefix and not args.prompt)

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

    data = gen_ctx_vectors(rows, encoder, tensorizer, args, True) 

    file = args.out_file + '-' + str(args.shard_id) + '.pkl'
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info('Writing results to %s' % file)
    with open(file, mode='wb') as f:
        pickle.dump(data, f)

    logger.info('Total passages processed %d. Written to %s', len(data), file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)
    add_tuning_params(parser)

    parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file')
    parser.add_argument('--out_file', required=True, type=str, default=None,
                        help='output file path to write results to')
    parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
    parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)
    
    
    main(args)
