source scripts/config.sh

python3 -m colbert.data_prep \
  --dataset $dataset \
  --split "test" \
  --collection $COLLECTION \
  --queries $QUERIES \
  --data_dir $DATA_DIR