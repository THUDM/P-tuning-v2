model_cfg=bert-base-uncased
filename=ptv2-dpr-multidata-128-40-8e-3-64
checkpoint=dpr_biencoder.39.1045
psl=64

topic=$1
topk=$2

python3 generate_dense_embeddings.py \
	--pretrained_model_cfg $model_cfg \
	--model_file checkpoints/$filename/$checkpoint \
	--shard_id 0 --num_shards 1 \
	--ctx_file data/oagqa-topic-v2/$topic-papers-10k.tsv \
	--out_file encoded_files/encoded-oagqa-$topic-$filename \
	--batch_size 128 \
	--sequence_length 256 \
	--prefix --pre_seq_len $psl 

python3 dense_retriever.py \
	--pretrained_model_cfg $model_cfg \
	--model_file checkpoints/$filename/$checkpoint \
	--ctx_file data/oagqa-topic-v2/$topic-papers-10k.tsv \
	--qa_file data/oagqa-topic-v2/$topic-questions.tsv \
	--encoded_ctx_file "encoded_files/encoded-oagqa-$topic-$filename-*.pkl" \
	--n-docs $topk \
	--batch_size 128 \
	--sequence_length 256 \
	--oagqa \
	--prefix --pre_seq_len $psl 
