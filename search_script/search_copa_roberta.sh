export TASK_NAME=superglue
export DATASET_NAME=copa
export CUDA_VISIBLE_DEVICES=8


for lr in 5e-3 7e-3 1e-2 
do
  for psl in 4 8 16 32 64 128
  do 
    for epoch in 20 40 60 80 100 120
    do
     python3 run.py \
        --model_name_or_path roberta-large \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --pre_seq_len $psl \
        --output_dir checkpoints/$DATASET_NAME-roberta-search/$DATASET_NAME-$epoch-$lr-$psl/ \
        --overwrite_output_dir \
        --hidden_dropout_prob 0.1 \
        --seed 11 \
        --save_strategy no \
        --evaluation_strategy epoch \
        --prefix
    done
  done
done

python3 search.py $DATASET_NAME roberta