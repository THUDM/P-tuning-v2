model_cfg=bert-base-uncased
bs=128
epoch=40
lr=1e-4
filename=ad-dpr-multidata-$bs-$epoch-$lr

python3 train_dense_encoder.py \
	--pretrained_model_cfg $model_cfg \
	--train_file "data/retriever/*-train.json" \
	--dev_file "data/retriever/*-dev.json" \
	--output_dir checkpoints/$filename \
	--seed 12345 \
	--do_lower_case \
	--max_grad_norm 2.0 \
	--sequence_length 256 \
	--warmup_percentage 0.05 \
	--val_av_rank_start_epoch 30 \
	--batch_size $bs \
	--learning_rate $lr \
	--num_train_epochs $epoch \
	--dev_batch_size $bs \
	--hard_negatives 1 \
	--adapter
