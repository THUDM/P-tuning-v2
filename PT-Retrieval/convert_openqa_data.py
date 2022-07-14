import json
import os
import sys

# dataset = sys.argv[1]

key = {
    "webq": "psg_id",
    "nq": "passage_id",
    "trivia": "psg_id",
    "curatedtrec": "psg_id"
}

def convert(dataset):
    corpus = []
    queries = []
    qrels = []

    data = json.load(open(f"data/retriever/{dataset}-dev.json", 'r'))

    for i, q in enumerate(data):
        qid = f"question-{i}"
        query = {
            "_id": qid,
            "text": q["question"]
        }
        for p in q["positive_ctxs"] +  q["negative_ctxs"] + q["hard_negative_ctxs"]:
            psg = {
                "_id": p[key[dataset]],
                "title": p["title"],
                "text": p["text"]
            }
            corpus.append(psg)

        for p in q["positive_ctxs"]:
            pid = p[key[dataset]]
            qrels.append(f"{qid}\t{pid}\t{1}")
        queries.append(query)


    dataset = f"openqa-{dataset}"

    data_dir = f"beir_eval/datasets/{dataset}"

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        os.mkdir(data_dir+"/qrels")

    with open(os.path.join(data_dir, "corpus.jsonl"), 'w') as f:
        for psg in corpus:
            f.write(json.dumps(psg) + '\n')

    with open(os.path.join(data_dir, "queries.jsonl"), 'w') as f:
        for query in queries:
            f.write(json.dumps(query) + '\n')

    with open(os.path.join(data_dir, "qrels", "test.tsv"), 'w') as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for rel in qrels:
            f.write(rel + '\n')

