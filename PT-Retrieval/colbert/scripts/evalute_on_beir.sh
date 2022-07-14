source scripts/config.sh

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

################################################################
# 0. BEIR Data Formatting: Format BEIR data useful for ColBERT #
################################################################ 

python3 -m colbert.data_prep \
  --dataset $dataset \
  --split $SPLIT \
  --collection $COLLECTION \
  --queries $QUERIES \
  --data_dir $DATA_DIR

############################################################################
# 1. Indexing: Encode document (token) embeddings using ColBERT checkpoint #
############################################################################ 

python3 -m torch.distributed.launch --master_port=$MASTER_PORT \
  --nproc_per_node=2 -m colbert.index \
  --root $OUTPUT_DIR \
  --doc_maxlen 300 \
  --mask-punctuation \
  --bsize 128 \
  --amp \
  --checkpoint $CHECKPOINT \
  --index_root $INDEX_ROOT \
  --index_name $INDEX_NAME \
  --collection $COLLECTION \
  --experiment $EXPERIMENT \
  --prefix --pre_seq_len $psl --similarity $SIMILARITY

###########################################################################################
# 2. Faiss Indexing (End-to-End Retrieval): Store document (token) embeddings using Faiss #
########################################################################################### 

python3 -m colbert.index_faiss \
  --index_root $INDEX_ROOT \
  --index_name $INDEX_NAME \
  --partitions $NUM_PARTITIONS \
  --sample 0.3 \
  --root $OUTPUT_DIR \
  --experiment $EXPERIMENT

####################################################################################
# 3. Retrieval: retrieve relevant documents of queries from faiss index checkpoint #
####################################################################################

python3 -m colbert.retrieve \
  --amp \
  --doc_maxlen 300 \
  --mask-punctuation \
  --bsize 256 \
  --queries $QUERIES \
  --nprobe 32 \
  --partitions $NUM_PARTITIONS \
  --faiss_depth 100 \
  --depth 100 \
  --index_root $INDEX_ROOT \
  --index_name $INDEX_NAME \
  --checkpoint $CHECKPOINT \
  --root $OUTPUT_DIR \
  --experiment $EXPERIMENT \
  --ranking_dir $RANKING_DIR \
  --prefix --pre_seq_len $psl --similarity $SIMILARITY


######################################################################
# 4. BEIR Evaluation: Evaluate Rankings with BEIR Evaluation Metrics #
######################################################################

python3 -m colbert.beir_eval \
  --dataset ${dataset} \
  --split $SPLIT \
  --collection $COLLECTION \
  --rankings "${RANKING_DIR}/ranking.tsv" \
  --data_dir $DATA_DIR \