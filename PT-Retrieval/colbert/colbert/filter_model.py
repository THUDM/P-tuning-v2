


import os
import random
import torch

from colbert.utils.parser import Arguments
from colbert.utils.runs import Run

from colbert.evaluation.loaders import load_colbert, load_qrels, load_queries
from colbert.indexing.faiss import get_faiss_index_name
from colbert.ranking.retrieval import retrieve
from colbert.ranking.batch_retrieval import batch_retrieve
from colbert.utils.utils import print_message, save_checkpoint
from colbert.utils.utils import print_message, load_checkpoint
from transformers import BertConfig

from colbert.parameters import DEVICE
from colbert.modeling.colbert import ColBERT
from colbert.modeling.prefix import PrefixColBERT
from colbert.utils.utils import print_message, load_checkpoint
from transformers import AdamW
from collections import OrderedDict, defaultdict

def main():
    random.seed(12345)

    parser = Arguments(description='End-to-end retrieval and ranking with ColBERT.')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_ranking_input()
    parser.add_retrieval_input()

    parser.add_argument('--faiss_name', dest='faiss_name', default=None, type=str)
    parser.add_argument('--faiss_depth', dest='faiss_depth', default=1024, type=int)
    parser.add_argument('--part-range', dest='part_range', default=None, type=str)
    parser.add_argument('--batch', dest='batch', default=False, action='store_true')
    parser.add_argument('--depth', dest='depth', default=1000, type=int)
    parser.add_argument('--ranking_dir', dest='ranking_dir', default=None, type=str)

    args = parser.parse()

    args.depth = args.depth if args.depth > 0 else None

    if args.part_range:
        part_offset, part_endpos = map(int, args.part_range.split('..'))
        args.part_range = range(part_offset, part_endpos)

    # colbert, checkpoint = load_colbert(args)
    if args.prefix:
        config = BertConfig.from_pretrained('bert-base-uncased', cache_dir=".cache")
        config.pre_seq_len = args.pre_seq_len
        config.prefix_hidden_size = args.prefix_hidden_size
        config.prefix_mlp = args.prefix_mlp
        colbert = PrefixColBERT.from_pretrained('bert-base-uncased', config=config,
                                          query_maxlen=args.query_maxlen,
                                          doc_maxlen=args.doc_maxlen,
                                          dim=args.dim,
                                          similarity_metric=args.similarity,
                                          mask_punctuation=args.mask_punctuation)
    else:
        config = BertConfig.from_pretrained('bert-base-uncased', cache_dir=".cache")
        colbert = ColBERT.from_pretrained('bert-base-uncased',
                                        config=config,
                                        query_maxlen=args.query_maxlen,
                                        doc_maxlen=args.doc_maxlen,
                                        dim=args.dim,
                                        similarity_metric=args.similarity,
                                        mask_punctuation=args.mask_punctuation)

    colbert = colbert.to(DEVICE)
    pretrain_state_dict = OrderedDict()
    for k, v in colbert.state_dict().items():
        pretrain_state_dict[k] = torch.clone(v)
    print(filter(lambda p: p.requires_grad, colbert.parameters()))
    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=1e-3, eps=1e-8)
    checkpoint = load_checkpoint(args.checkpoint, colbert, optimizer=optimizer)

    for k, v in pretrain_state_dict.items():
        print(k)
        print(bool(torch.all(v == colbert.state_dict().get(k))))
    print(args.checkpoint)
    print(checkpoint['arguments'])
    name = "checkpoints/ptv2.colbert.msmarco.cosine.1e-3.64.filtered/colbert-360000.dnn"
    save_checkpoint(name, 0, 360000, colbert, optimizer, checkpoint['arguments'], True)


if __name__ == "__main__":
    main()
