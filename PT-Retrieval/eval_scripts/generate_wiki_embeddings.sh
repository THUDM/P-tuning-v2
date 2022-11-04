model_cfg=bert-base-uncased
filename=ptv2-dpr-multidata-128-40-8e-3-64
checkpoint=dpr_biencoder.39.1045
psl=64

run_func() {
CUDA_VISIBLE_DEVICES=$1 python3 generate_dense_embeddings.py \
	--pretrained_model_cfg $model_cfg \
	--model_file checkpoints/$filename/$checkpoint \
	--shard_id $2 --num_shards 8 \
	--ctx_file data/wikipedia_split/psgs_w100.tsv \
	--out_file encoded_files/encoded-wiki-$filename \
	--batch_size 128 \
	--sequence_length 256 \
	--prefix --pre_seq_len $psl 
}


for node in 0 1 2 3 4 5 6 7
do
	shard_id=$((node))
	run_func $node $shard_id &
done

wait
echo "All done"
