#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .hf_models import get_bert_biencoder_components, get_bert_reader_components, get_bert_tensorizer, get_roberta_tensorizer

"""
 'Router'-like set of methods for component initialization with lazy imports 
"""

ENCODER_INITIALIZERS = {
    "dpr": get_bert_biencoder_components,
}

READER_INITIALIZERS = {
    'bert': get_bert_reader_components,
}

TENSORIZER_INITIALIZERS = {
    'bert': get_bert_tensorizer,
    'roberta': get_roberta_tensorizer,
}

def init_encoder_components(encoder_type: str, args, **kwargs):
    if encoder_type in ENCODER_INITIALIZERS:
        return ENCODER_INITIALIZERS[encoder_type](args, **kwargs)
    else:
        raise RuntimeError('unsupported encoder type: {}'.format(encoder_type))

def init_reader_components(reader_type: str, args, **kwargs):
    if reader_type in READER_INITIALIZERS:
        return READER_INITIALIZERS[reader_type](args)
    else:
        raise RuntimeError('unsupported reader type: {}'.format(reader_type))

def init_tensorizer(tensorizer_type: str, args, **kwargs):
    if tensorizer_type in TENSORIZER_INITIALIZERS:
        return TENSORIZER_INITIALIZERS[tensorizer_type](args)
    else:
        raise RuntimeError('unsupported tensorizer type: {}'.format(tensorizer_type))
