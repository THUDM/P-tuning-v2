#!/bin/bash

export dataset=$1
export psl=64
export step=36
export SIMILARITY=cosine
export checkpoint=ptv2.colbert.msmarco.cosine.1e-3.64

export DATA_DIR=../beir_eval/datasets/$dataset/

export CHECKPOINT=./checkpoints/$checkpoint/colbert-${step}0000.dnn

# Path where preprocessed collection and queries are present
export COLLECTION="../beir_eval/preprocessed_data/${dataset}/collection.tsv"
export QUERIES="../beir_eval/preprocessed_data/${dataset}/queries.tsv"

# Path to store the faiss index and run output
export INDEX_ROOT="output/index"
export OUTPUT_DIR="output/output"
export EXPERIMENT=$checkpoint.$step.$dataset
# Path to store the rankings file
export RANKING_DIR="output/rankings/${checkpoint}/${step}/${dataset}"

# Num of partitions in for IVPQ Faiss index (You must decide yourself)
export NUM_PARTITIONS=96

# Some Index Name to store the faiss index
export INDEX_NAME=index.$checkpoint.$step.$dataset

if [ "$dataset" = "msmarco" ]; then
    SPLIT="dev"
else
    SPLIT="test"
fi

echo $SPLIT