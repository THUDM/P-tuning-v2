import logging
from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn

from transformers import AdamW, BertConfig, BertModel, BertTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from dpr.models.hf_models import get_optimizer, get_bert_tensorizer
from dpr.utils.data_utils import Tensorizer
from .ledreader import LedReader

logger = logging.getLogger(__name__)

def get_led_reader_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, 'dropout') else 0.0
    # encoder = HFSeq2SeqLM.init_encoder(args.pretrained_model_cfg,
    #                                      projection_dim=args.projection_dim, dropout=dropout)
    print("led")
    cfg = AutoConfig.from_pretrained(args.pretrained_model_cfg)
    if dropout != 0:
        cfg.attention_dropout = dropout
        cfg.gradient_checkpointing = False
        cfg.use_cache=False
        cfg.attention_window = [args.attention_window] * len(cfg.attention_window)
    encoder = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_cfg, config=cfg)

    hidden_size = encoder.config.hidden_size
    reader = LedReader(encoder, hidden_size)
    # for param in reader.parameters():
    #     print(param.data.shape)
    optimizer = get_optimizer(reader,
                              learning_rate=args.learning_rate,
                              adam_eps=args.adam_eps, weight_decay=args.weight_decay,
                              ) if not inference_only else None

    # tensorizer = get_bert_tensorizer(args, AutoTokenizer.from_pretrained(args.pretrained_model_cfg))
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_cfg, add_special_tokens=False)
    return tokenizer, reader, optimizer

        
class HFSeq2SeqLM(AutoModelForSeq2SeqLM):

    def __init__(self, config, project_dim: int = 0):
        AutoModelForSeq2SeqLM.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs) -> BertModel:
        cfg = AutoConfig.from_pretrained(cfg_name)
        if dropout != 0:
            cfg.attention_dropout = dropout
            cfg.gradient_checkpointing = False
            cfg.attention_window = [1024] * len(cfg.attention_window)
            # config.attention_window = [attention_window_size] * len(config.attention_window)
            # config.gradient_checkpointing = gradient_checkpointing
        return cls.from_pretrained(cfg_name, config=cfg, **kwargs)

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(input_ids=input_ids,
                                                                            token_type_ids=token_type_ids,
                                                                            attention_mask=attention_mask)
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(input_ids=input_ids, token_type_ids=token_type_ids,
                                                             attention_mask=attention_mask)

        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

