
model_cfg=bert-base-uncased
filename=ptv2-dpr-multidata-128-40-8e-3-64
checkpoint=dpr_biencoder.39.1045
psl=64
dir=oagqa_eval_lists/*

for f in $dir
do
    echo $f
    echo ${f#*/}
	python3 batch_dense_retrieval.py \
		--pretrained_model_cfg $model_cfg \
		--model_file checkpoints/$filename/$checkpoint \
		--prefix --pre_seq_len $psl \
		--ctx_dir data/oagqa-topic-v2 \
		--out_dir encoded_oagqa_files \
		--input_file $f \
		--result_file "results/ptv2-dpr/oagqa_results_${f#*/}.json" \
		--batch_size 128 \
		--sequence_length 256 \
		--oagqa \
		--n-docs 20 
done

