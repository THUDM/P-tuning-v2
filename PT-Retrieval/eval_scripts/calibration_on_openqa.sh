model_cfg=bert-base-uncased
filename=ptv2-dpr-multidata-128-40-8e-3-64
checkpoint=dpr_biencoder.39.1045
psl=64

dataset=$1

echo "Ploy Calibration Curve"
python3 calibration/calibration_plot_openqa.py \
	--pretrained_model_cfg $model_cfg \
	--model_file ./checkpoints/$filename/$checkpoint \
	--batch_size 128 \
	--sequence_length 512 \
    --prefix --pre_seq_len $psl \
    --dataset $dataset

echo "Calculate ECE"
python3 calibration/calibration_ece_openqa.py \
	--pretrained_model_cfg $model_cfg \
	--model_file checkpoints/$filename/$checkpoint \
	--ctx_file data/wikipedia_split/psgs_w100.tsv \
	--encoded_ctx_file "encoded_files/encoded-wiki-$filename-*.pkl" \
	--n-docs 5 \
	--batch_size 128 \
	--sequence_length 256 \
	--prefix --pre_seq_len $psl \
	--dataset $dataset