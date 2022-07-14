#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Command line arguments utils
"""

import argparse
import logging
import os
import random
import socket

import numpy as np
import torch

logger = logging.getLogger()


def add_tokenizer_params(parser: argparse.ArgumentParser):
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")


def add_encoder_params(parser: argparse.ArgumentParser):
    """
        Common parameters to initialize an encoder-based model
    """
    parser.add_argument("--pretrained_model_cfg", default=None, type=str, help="config name for model initialization")
    parser.add_argument("--encoder_model_type", default=None, type=str,
                        help="model type. One of [hf_bert, pytext_bert, fairseq_roberta]")
    parser.add_argument('--pretrained_file', type=str, help="Some encoders need to be initialized from a file")
    parser.add_argument("--model_file", default=None, type=str,
                        help="Saved bi-encoder checkpoint file to initialize the model")
    parser.add_argument("--projection_dim", default=0, type=int,
                        help="Extra linear layer on top of standard bert/roberta encoder")
    parser.add_argument("--sequence_length", type=int, default=512, help="Max length of the encoder input sequence")

def add_tuning_params(parser: argparse.ArgumentParser):
    # p-tuning v2 params
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument("--pre_seq_len", type=int, default=8)
    parser.add_argument("--prefix_hidden_size", type=int, default=512)
    parser.add_argument("--prefix_mlp", action="store_true")

    parser.add_argument("--bitfit", action="store_true")

    # adapter params
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--adapter_config", type=str, default="pfeiffer")

    # p-tuning params
    parser.add_argument("--prompt", action="store_true")
    
def add_training_params(parser: argparse.ArgumentParser):
    """
        Common parameters for training
    """
    add_cuda_params(parser)
    parser.add_argument("--train_file", default=None, type=str, help="File pattern for the train set")
    parser.add_argument("--dev_file", default=None, type=str, help="")
    parser.add_argument("--tb_folder", type=str)

    parser.add_argument("--batch_size", default=2, type=int, help="Amount of questions per batch")
    parser.add_argument("--dev_batch_size", type=int, default=4,
                        help="amount of questions per batch for dev set validation")
    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization and dataset shuffling")

    parser.add_argument("--adam_eps", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default='(0.9, 0.999)', type=str, help="Betas for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--log_batch_step", default=100, type=int, help="")
    parser.add_argument("--train_rolling_loss_step", default=100, type=int, help="")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="")
    parser.add_argument("--momentum", default=0.9, type=float, help="")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")

    parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout", default=0.1, type=float, help="")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--ctx_field', type=str, choices=['title', 'abstract', 'text'], default='text')
    

def add_cuda_params(parser: argparse.ArgumentParser):
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")


def add_reader_preprocessing_params(parser: argparse.ArgumentParser):
    parser.add_argument("--gold_passages_src", type=str,
                        help="File with the original dataset passages (json format). Required for train set")
    parser.add_argument("--gold_passages_src_dev", type=str,
                        help="File with the original dataset passages (json format). Required for dev set")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="number of parallel processes to binarize reader data")




def get_encoder_checkpoint_params_names():
    return ['do_lower_case', 'pretrained_model_cfg', 'encoder_model_type',
            'pretrained_file',
            'projection_dim', 'sequence_length']


def get_encoder_params_state(args):
    """
     Selects the param values to be saved in a checkpoint, so that a trained model faile can be used for downstream
     tasks without the need to specify these parameter again
    :return: Dict of params to memorize in a checkpoint
    """
    params_to_save = get_encoder_checkpoint_params_names()

    r = {}
    for param in params_to_save:
        r[param] = getattr(args, param)
    return r


def set_encoder_params_from_state(state, args):
    if not state:
        return
    params_to_save = get_encoder_checkpoint_params_names()

    override_params = [(param, state[param]) for param in params_to_save if param in state and state[param]]
    for param, value in override_params:
        if hasattr(args, param):
            logger.warning('Overriding args parameter value from checkpoint state. Param = %s, value = %s', param,
                           value)
        setattr(args, param, value)
    return args


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_args_gpu(args):
    """
     Setup arguments CUDA, GPU & distributed training
    """

    if args.local_rank == -1 or args.no_cuda:  # single-node multi-gpu (or cpu) mode
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    ws = os.environ.get('WORLD_SIZE')

    args.distributed_world_size = int(ws) if ws else 1

    logger.info(
        'Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d', socket.gethostname(),
        args.local_rank, device,
        args.n_gpu,
        args.distributed_world_size)
    logger.info("16-bits training: %s ", args.fp16)


def print_args(args):
    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")
