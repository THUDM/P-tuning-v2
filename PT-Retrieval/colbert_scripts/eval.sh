export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=.

model_cfg=bert-base-uncased
psl=64
checkpoint=ptv2.colbert.msmarco.cosine.1e-3.64
filename=$checkpoint/colbert-360000.dnn

# python3 generate_dense_embeddings_colbert.py \
# 	--encoder_type dpr \
# 	--pretrained_model_cfg $model_cfg \
# 	--shard_id 0 --num_shards 1 \
# 	--ctx_file data/oagqa-topic-v2/geometry-papers-10k.tsv \
# 	--out_file encoded_colbert_files/encoded-geometry-$checkpoint \
# 	--batch_size 100 \
# 	--sequence_length 256 \
# 	--prefix \
# 	--pre_seq_len $psl \
#   --amp --doc_maxlen 300  --mask-punctuation --bsize 256 \
#   --checkpoint checkpoints/$filename

python3 dense_retriever_colbert.py \
	--encoder_type dpr \
	--pretrained_model_cfg $model_cfg \
	--ctx_file data/oagqa-topic-v2/geometry-papers-10k.tsv \
	--qa_file data/oagqa-topic-v2/geometry-questions.tsv \
	--encoded_ctx_file "encoded_colbert_files/encoded-geometry-$checkpoint-*.pkl" \
	--n-docs 100 \
	--batch_size 100 \
	--sequence_length 256 \
	--prefix \
	--pre_seq_len $psl \
    --amp --doc_maxlen 300  --mask-punctuation --bsize 256 \
    --checkpoint checkpoints/$filename \
    --bs 4 --oagqa