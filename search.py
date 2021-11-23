import json
import os
import sys

from glob import glob

from tasks.utils import *

TASK = sys.argv[1]
MODEL = sys.argv[2]

if len(sys.argv) == 4:
    METRIC = sys.argv[3]
elif TASK in GLUE_DATASETS + SUPERGLUE_DATASETS:
    METRIC = "accuracy"
elif TASK in NER_DATASETS + SRL_DATASETS + QA_DATASETS:
    METRIC = "f1"

best_score = 0

files = glob(f"./checkpoints/{TASK}-{MODEL}-search/*/best_results.json")

for f in files:
    metrics = json.load(open(f, 'r'))
    if metrics["best_eval_"+METRIC] > best_score:
        best_score = metrics["best_eval_"+METRIC]
        best_metrics = metrics
        best_file_name = f

print(f"best_{METRIC}: {best_score}")
print(f"best_metrics: {best_metrics}")
print(f"best_file: {best_file_name}")

