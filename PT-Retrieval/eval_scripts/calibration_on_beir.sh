model_cfg=bert-base-uncased
filename=ptv2-dpr-multidata-128-40-8e-3-64
checkpoint=dpr_biencoder.39.1045
psl=64

dataset=$1

if [ "$dataset" = "msmarco" ]; then
    split="dev"
else
    split="test"
fi

python3 calibration/calibration_beir.py \
	--pretrained_model_cfg $model_cfg \
	--model_file ./checkpoints/$filename/$checkpoint \
	--batch_size 128 \
	--sequence_length 512 \
    --prefix --pre_seq_len $psl \
    --dataset $dataset --split $split

