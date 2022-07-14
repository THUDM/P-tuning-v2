export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=.

lr=1e-3
psl=180

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT colbert/train.py \
    --amp \
    --doc_maxlen 300 \
    --mask-punctuation \
    --lr $lr \
    --bsize 32 --accum 1 \
    --triples ../data/msmarco/triples.train.small.tsv \
    --root ./ \
    --experiment checkpoints \
    --similarity cosine \
    --run ptv2.colbert.msmarco.cosine.$lr.$psl \
    --prefix --pre_seq_len $psl