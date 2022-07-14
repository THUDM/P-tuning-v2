#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn
from transformers import AdamW, AutoConfig, BertConfig, BertModel, BertTokenizer, RobertaTokenizer
from .prefix import BertPrefixModel, BertPromptModel

from dpr.utils.data_utils import Tensorizer
from .biencoder import BiEncoder
from .reader import Reader

logger = logging.getLogger(__name__)

def get_bert_biencoder_components(args, inference_only: bool = False, **kwargs):  
    if args.prefix:
        logger.info("***** P-tuning v2 mode *****")
        question_encoder = BertPrefixEncoder.init_encoder(args, **kwargs)
        ctx_encoder = BertPrefixEncoder.init_encoder(args, **kwargs)
    elif args.bitfit:
        logger.info("***** Bitfit mode *****")
        question_encoder = BertEncoder.init_encoder(args, **kwargs)
        ctx_encoder = BertEncoder.init_encoder(args, **kwargs)
    elif args.adapter:
        logger.info("***** Adapter mode *****")
        question_encoder = BertAdapterEncoder.init_encoder(args, **kwargs)
        ctx_encoder = BertAdapterEncoder.init_encoder(args, **kwargs)
    elif args.prompt:
        logger.info("***** Prompt (Lester at el. & P-tuning) mode *****")
        question_encoder = BertPromptEncoder.init_encoder(args, **kwargs)
        ctx_encoder = BertPromptEncoder.init_encoder(args, **kwargs)
    else:
        logger.info("***** Finetuning mode *****")
        question_encoder = BertEncoder.init_encoder(args, **kwargs)
        ctx_encoder = BertEncoder.init_encoder(args, **kwargs)


    biencoder = BiEncoder(question_encoder, ctx_encoder)

    if args.bitfit:
        _deactivate_relevant_gradients(biencoder,trainable_components=["bias"])

    question_encoder.print_params()

    optimizer = get_optimizer(biencoder,
                              learning_rate=args.learning_rate,
                              adam_eps=args.adam_eps, weight_decay=args.weight_decay,
                              ) if not inference_only else None
    
    tensorizer = get_bert_tensorizer(args)

    return tensorizer, biencoder, optimizer

def get_bert_reader_components(args, inference_only: bool = False, **kwargs):
    encoder = BertEncoder.init_encoder(args)

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = get_optimizer(reader,
                              learning_rate=args.learning_rate,
                              adam_eps=args.adam_eps, weight_decay=args.weight_decay,
                              ) if not inference_only else None

    tensorizer = get_bert_tensorizer(args)
    return tensorizer, reader, optimizer

def get_bert_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_bert_tokenizer(args.pretrained_model_cfg, do_lower_case=args.do_lower_case)
    return BertTensorizer(tokenizer, args.sequence_length)


def get_roberta_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_roberta_tokenizer(args.pretrained_model_cfg, do_lower_case=args.do_lower_case)
    return RobertaTensorizer(tokenizer, args.sequence_length)


def get_optimizer(model: nn.Module, learning_rate: float = 1e-5, adam_eps: float = 1e-8,
                  weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer

def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case, cache_dir=".cache")


def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return RobertaTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case, cache_dir=".cache")

class BertEncoder(BertModel):

    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, args, **kwargs) -> BertModel:
        cfg = BertConfig.from_pretrained(args.pretrained_model_cfg, cache_dir=".cache")
        dropout = args.dropout if hasattr(args, 'dropout') else 0.0
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(args.pretrained_model_cfg, config=cfg, cache_dir=".cache", **kwargs)

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(input_ids=input_ids,
                                                                            token_type_ids=token_type_ids,
                                                                            attention_mask=attention_mask,return_dict=False)
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(input_ids=input_ids, token_type_ids=token_type_ids,
                                                             attention_mask=attention_mask,return_dict=False)
        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

    def print_params(self):
        all_param = 0
        trainable_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_param += param.numel()
                
        logger.info( " ***************** PARAMETERS ***************** ")
        logger.info(f" # all param       : {all_param}")
        logger.info(f" # trainable param : {trainable_param}")
        logger.info( " ***************** PARAMETERS ***************** ")

class BertPrefixEncoder(BertPrefixModel):

    def __init__(self, config, project_dim: int = 0):
        BertPrefixModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, args, **kwargs) -> BertPrefixModel:

        cfg = AutoConfig.from_pretrained(args.pretrained_model_cfg, cache_dir=".cache")

        cfg.pre_seq_len = args.pre_seq_len
        cfg.prefix_mlp = args.prefix_mlp
        cfg.prefix_hidden_size = args.prefix_hidden_size

        dropout = args.dropout if hasattr(args, 'dropout') else 0.0
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        return cls.from_pretrained(args.pretrained_model_cfg, config=cfg, cache_dir=".cache", **kwargs)

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(input_ids=input_ids,
                                                                            token_type_ids=token_type_ids,
                                                                            attention_mask=attention_mask,return_dict=False)
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(input_ids=input_ids, token_type_ids=token_type_ids,
                                                             attention_mask=attention_mask,return_dict=False)
        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

    def print_params(self):
        all_param = 0
        trainable_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_param += param.numel()
                
        logger.info( " ***************** PARAMETERS ***************** ")
        logger.info(f" # all param       : {all_param}")
        logger.info(f" # trainable param : {trainable_param}")
        logger.info( " ***************** PARAMETERS ***************** ")

class BertAdapterEncoder(BertModel):

    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, args, **kwargs):
        cfg = AutoConfig.from_pretrained(args.pretrained_model_cfg, cache_dir=".cache")

        cfg.pre_seq_len = args.pre_seq_len
        cfg.prefix_mlp = args.prefix_mlp
        cfg.prefix_hidden_size = args.prefix_hidden_size

        dropout = args.dropout if hasattr(args, 'dropout') else 0.0
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        model = cls.from_pretrained(args.pretrained_model_cfg, config=cfg, cache_dir=".cache", **kwargs)
        model.add_adapter("adapter", config=args.adapter_config)
        model.train_adapter("adapter")

        return model
    
    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(input_ids=input_ids,
                                                                            token_type_ids=token_type_ids,
                                                                            attention_mask=attention_mask,return_dict=False)
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(input_ids=input_ids, token_type_ids=token_type_ids,
                                                             attention_mask=attention_mask,return_dict=False)
        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

    def print_params(self):
        all_param = 0
        trainable_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_param += param.numel()
                
        logger.info( " ***************** PARAMETERS ***************** ")
        logger.info(f" # all param       : {all_param}")
        logger.info(f" # trainable param : {trainable_param}")
        logger.info( " ***************** PARAMETERS ***************** ")

class BertPromptEncoder(BertPromptModel):

    def __init__(self, config, project_dim: int = 0):
        BertPromptModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, args, **kwargs) -> BertPromptModel:

        cfg = AutoConfig.from_pretrained(args.pretrained_model_cfg, cache_dir=".cache")

        cfg.pre_seq_len = args.pre_seq_len
        cfg.output_hidden_states = False

        dropout = args.dropout if hasattr(args, 'dropout') else 0.0
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        return cls.from_pretrained(args.pretrained_model_cfg, config=cfg, cache_dir=".cache", **kwargs)

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[T, ...]:
        hidden_states = None
        sequence_output, pooled_output = super().forward(input_ids=input_ids, token_type_ids=token_type_ids,
                                                            attention_mask=attention_mask,return_dict=False)
        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

    def print_params(self):
        all_param = 0
        trainable_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_param += param.numel()
                
        logger.info( " ***************** PARAMETERS ***************** ")
        logger.info(f" # all param       : {all_param}")
        logger.info(f" # trainable param : {trainable_param}")
        logger.info( " ***************** PARAMETERS ***************** ")


class BertTensorizer(Tensorizer):
    def __init__(self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(title, text_pair=text, add_special_tokens=add_special_tokens,
                                              max_length=self.max_length,
                                              pad_to_max_length=False, truncation='only_second')#, truncation='longest_first')
        else:
            token_ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens, max_length=self.max_length,
                                              pad_to_max_length=False, truncation='only_second')#, truncation='longest_first')

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_type_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad


class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(tokenizer, max_length, pad_to_max=pad_to_max)


def _deactivate_relevant_gradients(model, trainable_components):
    # turns off the model parameters requires_grad except the trainable bias terms.
    for param in model.parameters():
        param.requires_grad = False
    if trainable_components:
        trainable_components = trainable_components + ['pooler.dense.bias']

    for name, param in model.named_parameters():
        for component in trainable_components:
            if component in name:
                param.requires_grad = True
                break
